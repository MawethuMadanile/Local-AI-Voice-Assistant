"""
stt/vad.py — Voice Activity Detection using Silero VAD.

Silero VAD:
  - Cross-platform (Windows / Linux / macOS)
  - CPU-only (no CUDA required)
  - Accepts 16 kHz int16 audio chunks
  - Actively maintained
  - pip install silero-vad

Usage
-----
    from stt.vad import VADDetector

    def on_speech(samples: np.ndarray):
        # samples: float32, 16 kHz, shape (N,)
        ...

    vad = VADDetector(on_speech_end=on_speech)
    mic = MicrophoneStream(16000, chunk_ms=32, on_chunk=vad.process_chunk)
    mic.start()
"""

import collections
import threading
import time
from typing import Callable

import numpy as np
import torch

from config import cfg


class VADDetector:
    """
    Wraps Silero VAD to detect speech and emit complete utterances.

    Parameters
    ----------
    on_speech_end : callable
        Called with a float32 numpy array (16 kHz) when speech ends.
    """

    def __init__(self, on_speech_end: Callable[[np.ndarray], None]) -> None:
        self._on_speech_end = on_speech_end
        self._sr = cfg.vad.sample_rate
        self._threshold = cfg.vad.threshold
        self._silence_chunks = int(
            cfg.vad.silence_duration * 1000 / cfg.vad.chunk_ms
        )
        self._min_speech_chunks = int(
            cfg.vad.min_speech_duration * 1000 / cfg.vad.chunk_ms
        )

        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        self._model.eval()

        self._voiced: list[bytes] = []
        self._silence_count = 0
        self._speaking = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def process_chunk(self, raw: bytes) -> None:
        """
        Feed one int16 PCM chunk (16 kHz mono).
        Thread-safe — called from the sounddevice audio thread.
        """
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples)

        with torch.no_grad():
            prob = float(self._model(tensor, self._sr).item())

        is_speech = prob >= self._threshold

        with self._lock:
            if is_speech:
                if not self._speaking:
                    self._speaking = True
                    self._silence_count = 0
                self._voiced.append(raw)
                self._silence_count = 0

            else:
                if self._speaking:
                    self._voiced.append(raw)
                    self._silence_count += 1

                    if self._silence_count >= self._silence_chunks:
                        self._flush()

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Emit the buffered utterance (must be called under self._lock)."""
        if len(self._voiced) >= self._min_speech_chunks:
            joined = b"".join(self._voiced)
            arr = np.frombuffer(joined, dtype=np.int16).astype(np.float32) / 32768.0
            # Fire callback in a separate thread so we don't block audio I/O
            t = threading.Thread(target=self._on_speech_end, args=(arr,), daemon=True)
            t.start()

        self._voiced = []
        self._speaking = False
        self._silence_count = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Discard any buffered audio (call when resuming after TTS playback)."""
        with self._lock:
            self._voiced = []
            self._speaking = False
            self._silence_count = 0
        self._model.reset_states()
