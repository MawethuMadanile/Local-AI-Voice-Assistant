"""
tts/synthesiser.py — Text-to-speech using Kokoro (ONNX runtime).

Voice options (English):
    bm_george   male, British   ← Jarvis-like (default)
    bm_lewis    male, British
    am_michael  male, American
    am_adam     male, American
    af_sarah    female, American
    bf_emma     female, British
"""

from __future__ import annotations

import re
import numpy as np
from typing import Iterator

from config import cfg


class Synthesiser:
    """
    Wraps kokoro_onnx.Kokoro to produce float32 audio arrays.
    Streams sentence-by-sentence to keep first-word latency low.
    """

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro
        self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        self._voice = cfg.tts.voice
        self._speed = cfg.tts.speed
        self._sr = cfg.tts.sample_rate

    # ------------------------------------------------------------------
    def synthesise(self, text: str, is_last: bool = False) -> tuple[np.ndarray, int]:
        """
        Convert a plain text string to a float32 audio array.

        Parameters
        ----------
        is_last : whether this is the final chunk in a stream. The tail-pad
                  is slightly longer on the last chunk so the final word
                  isn't clipped before playback ends.
        """
        if not text.strip():
            return np.zeros(1, dtype=np.float32), self._sr

        samples, sr = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang="en-us",
        )
        samples = np.array(samples, dtype=np.float32)

        if cfg.tts.trim_silence:
            # Only trim the leading silence — never the tail.
            # Trimming the tail clips the last word of every chunk.
            samples = _trim_leading_silence(samples, sr, top_db=cfg.tts.trim_top_db)

        # Pad the tail so the final word fully decays before playback stops.
        # Last chunk gets a longer pad (300 ms) so the sentence ending
        # doesn't feel cut off; mid-chunks get a shorter pad (80 ms) to
        # keep the flow tight without an audible gap between sentences.
        pad_ms = 120 if is_last else 20
        pad_samples = int(sr * pad_ms / 1000)
        samples = np.concatenate([samples, np.zeros(pad_samples, dtype=np.float32)])

        return samples, sr

    # ------------------------------------------------------------------
    def stream(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        """
        Yield audio chunks sentence by sentence.

        Splits on sentence boundaries first (preserving punctuation for
        correct pacing), then strips punctuation before synthesis so
        Kokoro receives clean input.
        """
        chunks = _chunk_text(text)
        # Filter empty chunks before we know the index of the last one
        chunks = [c for c in chunks if c.strip()]

        for i, chunk in enumerate(chunks):
            spoken = _strip_punctuation(chunk)
            if not spoken.strip():
                continue
            is_last = (i == len(chunks) - 1)
            audio, sr = self.synthesise(spoken, is_last=is_last)
            yield audio, sr


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _strip_punctuation(text: str) -> str:
    """
    Remove all punctuation marks from text destined for speech synthesis.
    Word characters and whitespace are preserved so Kokoro receives clean,
    natural-sounding input without vocalising symbols.
    """
    return re.sub(r"[^\w\s]", "", text)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

# Kokoro sounds most natural when each synthesis call gets 1–3 short
# sentences.  Too short → choppy with audible joins.  Too long → high
# latency before first audio.  ~200 chars is the sweet spot.
MAX_CHUNK_CHARS = 200


def _chunk_text(text: str) -> list[str]:
    """
    Split text into natural spoken chunks.

    Strategy:
    1. Split on sentence-ending punctuation (before stripping it).
    2. Merge short consecutive sentences into one chunk up to MAX_CHUNK_CHARS.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= MAX_CHUNK_CHARS:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Silence trimming
# ---------------------------------------------------------------------------

def _trim_leading_silence(
    samples: np.ndarray,
    sr: int,
    top_db: float = 22.0,
) -> np.ndarray:
    """
    Trim only the leading silence from an audio array.

    The tail is left untouched — trimming it clips the last word of each
    chunk, which is exactly the stuttering/cutoff problem we're fixing.
    """
    frame_len = int(sr * 0.01)    # 10 ms frames
    threshold = 10 ** (-top_db / 20.0)

    energy = np.array([
        np.sqrt(np.mean(samples[i:i + frame_len] ** 2))
        for i in range(0, len(samples) - frame_len, frame_len)
    ])

    above = np.where(energy > threshold)[0]
    if len(above) == 0:
        return samples

    # Only move the start forward — leave the end alone
    start = above[0] * frame_len
    return samples[start:]