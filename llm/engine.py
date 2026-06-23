"""
llm/engine.py — LLM inference using llama-cpp-python (GGUF format).
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

from config import cfg, ROOT


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
        self._history: list[dict[str, str]] = []
        self._system = cfg.llm.system_prompt

        # Infinite cycle through the configured titles: Sir, Mr Madanile, Sir, ...
        self._titles = itertools.cycle(cfg.llm.user_titles)

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

    def chat(self, user_text: str) -> str:
        # Pick the next title for this turn
        title = next(self._titles)

        self._history.append({"role": "user", "content": user_text})

        prompt = self._build_prompt(title)

        result = self._llm(  # type: ignore
            prompt,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            stop=["</s>", "[INST]"],
            echo=False,
        )
        raw = result["choices"][0]["text"].strip()  # type: ignore
        reply = _sanitise(raw)

        self._history.append({"role": "assistant", "content": reply})

        max_messages = cfg.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

        return reply

    def _build_prompt(self, title: str) -> str:
        """
        Injects the current title into the system prompt so the model
        knows which form of address to use for this turn.
        """
        system_with_title = f"{self._system}\nAddress the user as '{title}' in this response."

        parts: list[str] = []
        for i, msg in enumerate(self._history):
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if i == 0:
                    parts.append(
                        f"[INST] <<SYS>>\n{system_with_title}\n<</SYS>>\n\n{content} [/INST] "
                    )
                else:
                    parts.append(f"[INST] {content} [/INST] ")
            else:
                parts.append(f"{content} </s>\n")

        return "".join(parts)

    def clear_history(self) -> None:
        self._history.clear()


def _sanitise(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.):\-]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\n{2,}", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()