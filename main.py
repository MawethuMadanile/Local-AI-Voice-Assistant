"""
main.py — ODIN (Offline Distributed Intelligence Network) voice assistant.

Architecture
------------

  MicrophoneStream (sounddevice)
        │  raw int16 PCM chunks (32 ms)
        ▼
  VADDetector (Silero VAD)
        │  on speech end: float32 numpy array
        ▼
  [utterance_queue]
        │
        ▼  (pipeline thread)
  Transcriber (faster-whisper / CTranslate2)
        │  plain text
        ▼
  LLMEngine (llama-cpp-python / GGUF)
        │  assistant reply text
        ▼
  Synthesiser (Kokoro ONNX)
        │  float32 audio + sample rate
        ▼
  play_audio (sounddevice)
"""

import queue
import sys
import threading
import time

import numpy as np

from audio import MicrophoneStream, play_audio, play_beep, stop_audio
from config import cfg
from llm import LLMEngine
from stt import Transcriber, VADDetector
from tts import Synthesiser


# ── Suppress Windows asyncio ConnectionResetError (WinError 10054) noise ──────
def _suppress_win_asyncio_noise() -> None:
    if sys.platform != "win32":
        return
    import asyncio

    def _handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError):
            return  # swallow silently
        loop.default_exception_handler(context)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.set_exception_handler(_handler)


_suppress_win_asyncio_noise()
# ─────────────────────────────────────────────────────────────────────────────

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
except Exception:
    pass

CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"
RESET  = "\033[0m"


class ODIN:
    def __init__(self) -> None:
        self._print_banner()
        self._log("Loading models…")

        self._utterance_q: queue.Queue[np.ndarray] = queue.Queue()
        self._playing = threading.Event()
        self._shutdown = threading.Event()

        self._log("  ► STT  (faster-whisper)…", end="")
        self._stt = Transcriber()
        print(" ✓")

        self._log("  ► LLM  (llama-cpp-python)…", end="")
        self._llm = LLMEngine()
        print(" ✓")

        self._log("  ► TTS  (Kokoro ONNX)…", end="")
        self._tts = Synthesiser()
        print(" ✓")

        self._vad = VADDetector(on_speech_end=self._on_speech_end)
        self._mic = MicrophoneStream(
            sample_rate=cfg.vad.sample_rate,
            chunk_ms=cfg.vad.chunk_ms,
            on_chunk=self._vad.process_chunk,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_speech_end(self, audio: np.ndarray) -> None:
        """Called by VAD (in audio thread) when an utterance is complete."""
        if not self._playing.is_set():
            self._utterance_q.put(audio)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _pipeline_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                audio = self._utterance_q.get(timeout=0.2)
            except queue.Empty:
                continue

            # 1. Transcribe
            text = self._stt.transcribe(audio)
            if not text:
                continue

            self._print_user(text)

            # 2. Generate reply
            try:
                reply = self._llm.chat(text)
            except Exception as exc:
                self._log(f"{RED}[LLM error] {exc}{RESET}")
                continue

            if not reply:
                continue

            # 3. Print and speak
            self._print_assistant(reply)
            self._speak(reply)

            # 4. Ready for next input
            play_beep()
            self._vad.reset()

    def _speak(self, text: str) -> None:
        """Synthesise and play TTS audio, blocking until done."""
        self._playing.set()
        try:
            for chunk, sr in self._tts.stream(text):
                if self._shutdown.is_set():
                    break
                play_audio(chunk, sr, blocking=True)
        finally:
            self._playing.clear()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        play_beep()
        self._log(f"\n{CYAN}Listening… (Ctrl+C to quit){RESET}\n")

        pipeline_thread = threading.Thread(
            target=self._pipeline_loop, daemon=True, name="pipeline"
        )
        pipeline_thread.start()

        self._mic.start()

        try:
            while not self._shutdown.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._log(f"\n{DIM}Shutting down…{RESET}")
        finally:
            self._shutdown.set()
            self._mic.stop()
            stop_audio()
            pipeline_thread.join(timeout=5)
            self._log("Goodbye.")

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, end: str = "\n") -> None:
        if cfg.verbose:
            print(msg, end=end, flush=True)

    def _print_user(self, text: str) -> None:
        print(f"{GREEN}You › {RESET}{text}")

    def _print_assistant(self, text: str) -> None:
        print(f"{YELLOW}ODIN › {RESET}{text}\n")

    def _print_banner(self) -> None:
        print(f"""
{CYAN}╔════════════════════════════════════════════╗
║                 O  D  I  N                 ║
║  Offline Distributed Intelligence Network  ║
╚════════════════════════════════════════════╝{RESET}
""")


if __name__ == "__main__":
    odin = ODIN()
    odin.run()