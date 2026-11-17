"""
llm/engine.py — LLM inference using llama-cpp-python (GGUF format).

Prompt format: Phi-3 chat template (<|system|> / <|user|> / <|assistant|>).
Conversation history is saved to data/history.json so Nova remembers
across restarts.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from config import cfg, ROOT

HISTORY_FILE = ROOT / "data" / "history.json"

# Phi-3 control tokens — stop generating the moment any of them appears
STOP_TOKENS = ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>", "</s>"]


class LLMEngine:
    def __init__(self) -> None:
        from llama_cpp import Llama

        model_path = self._resolve_model()

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=cfg.llm.context_length,
            n_gpu_layers=cfg.llm.n_gpu_layers,
            n_threads=cfg.llm.n_threads,
            n_batch=512,
            verbose=False,
        )
        self._system = cfg.llm.system_prompt
        self._history: list[dict[str, str]] = self._load_history()

        # Cycle through the configured titles; falls back to "Sir" if the
        # config has no user_titles field.
        titles = getattr(cfg.llm, "user_titles", None) or ["Sir"]
        self._titles = itertools.cycle(titles)

    # ------------------------------------------------------------------
    def _resolve_model(self) -> Path:
        local = ROOT / cfg.llm.model_path
        if local.exists():
            return local

        print(f"[LLM] Model not found at {local}. Downloading from HuggingFace…")
        try:
            from huggingface_hub import hf_hub_download
            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=cfg.llm.hf_repo,
                filename=cfg.llm.hf_filename,
                local_dir=str(local.parent),
            )
            return Path(downloaded)
        except Exception as e:
            raise RuntimeError(
                f"Could not find or download LLM model.\n"
                f"Place a .gguf file at {local}\n"
                f"or set cfg.llm.model_path.  Error: {e}"
            ) from e

    # ------------------------------------------------------------------
    def chat(self, user_text: str) -> str:
        title = next(self._titles)

        self._history.append({"role": "user", "content": user_text})
        prompt = self._build_prompt(title)

        result = self._llm(  # type: ignore
            prompt,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            stop=STOP_TOKENS,
            echo=False,
        )
        raw = result["choices"][0]["text"].strip()  # type: ignore
        reply = _sanitise(raw)

        if not reply:
            # Nothing speakable came back: drop the user turn so the
            # history stays a clean user/assistant alternation.
            self._history.pop()
            return ""

        self._history.append({"role": "assistant", "content": reply})

        max_messages = cfg.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
            # History must always open with a user turn
            while self._history and self._history[0]["role"] != "user":
                self._history.pop(0)

        self._save_history()
        return reply

    # ------------------------------------------------------------------
    def _build_prompt(self, title: str) -> str:
        """Build a Phi-3 chat prompt. The system turn is included on every
        call, so it is never lost when old history is trimmed."""
        system = f"{self._system} Address the user as {title} in this response."

        parts: list[str] = [f"<|system|>\n{system}<|end|>\n"]
        for msg in self._history:
            tag = "user" if msg["role"] == "user" else "assistant"
            parts.append(f"<|{tag}|>\n{msg['content']}<|end|>\n")

        # Cue the model to start its reply
        parts.append("<|assistant|>\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    def _load_history(self) -> list[dict[str, str]]:
        """Read the saved conversation; start fresh if missing or corrupt."""
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            history = [
                m for m in data
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            while history and history[0]["role"] != "user":
                history.pop(0)
            return history
        except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
            return []

    def _save_history(self) -> None:
        """Write atomically so a crash mid-write can't corrupt the file."""
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = HISTORY_FILE.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(HISTORY_FILE)
        except OSError:
            pass  # memory is a nice-to-have; never crash the assistant over it

    def clear_history(self) -> None:
        self._history.clear()
        self._save_history()


# ---------------------------------------------------------------------------
# Reply sanitisation
# ---------------------------------------------------------------------------

def _sanitise(text: str) -> str:
    # Phi-3 chat template tokens
    text = re.sub(r"<\|.*?\|>", "", text)
    # Code fences: closed blocks first, then an unclosed one (cut-off reply)
    text = re.sub(r"```[\w]*\n?.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*", "", text, flags=re.DOTALL)
    text = text.replace("`", "")
    # Markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Numbered list prefixes and bullet points
    text = re.sub(r"^\s*\d+[.):\-]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    # Bold / italic markers
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    # Bracketed stage directions
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{2,}", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()