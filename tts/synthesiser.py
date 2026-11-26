"""
tts/synthesiser.py — Text-to-speech using Kokoro (ONNX runtime).

Voice options (English):
    af_bella    female, American (current)
    af_sarah    female, American
    bf_emma     female, British
    bm_george   male, British
    bm_lewis    male, British
    am_michael  male, American
    am_adam     male, American
"""

from __future__ import annotations

import re
import numpy as np
from typing import Iterator

from config import cfg, ROOT

# Expand contractions ("I'm" -> "I am") before synthesis. Apostrophes are
# preserved either way; set this to False if the speech sounds too stiff.
EXPAND_CONTRACTIONS = True


class Synthesiser:
    """
    Wraps kokoro_onnx.Kokoro to produce float32 audio arrays.
    Streams chunk by chunk to keep first-word latency low.
    """

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro
        self._kokoro = Kokoro(
            str(ROOT / "kokoro-v1.0.onnx"),
            str(ROOT / "voices-v1.0.bin"),
        )
        self._voice = cfg.tts.voice
        self._speed = cfg.tts.speed
        self._sr = cfg.tts.sample_rate

    # ------------------------------------------------------------------
    def synthesise(self, text: str, is_last: bool = False) -> tuple[np.ndarray, int]:
        """
        Convert a plain text string to a float32 audio array.

        is_last: the final chunk gets a longer tail pad so the last word
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
            # Only trim leading silence. Trimming the tail clips the last
            # word of every chunk.
            samples = _trim_leading_silence(samples, sr, top_db=cfg.tts.trim_top_db)

        # Short pad between chunks, longer on the last one
        pad_ms = 120 if is_last else 20
        pad_samples = int(sr * pad_ms / 1000)
        samples = np.concatenate([samples, np.zeros(pad_samples, dtype=np.float32)])

        return samples, sr

    # ------------------------------------------------------------------
    def stream(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        """
        Yield audio chunks.

        Splits on sentence boundaries first (while punctuation is still
        there), then cleans each chunk before synthesis.
        """
        chunks = [c for c in _chunk_text(text) if c.strip()]

        for i, chunk in enumerate(chunks):
            spoken = _prepare_for_speech(chunk)
            if not spoken.strip():
                continue
            is_last = (i == len(chunks) - 1)
            audio, sr = self.synthesise(spoken, is_last=is_last)
            yield audio, sr


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "i'm": "I am",
    "i'll": "I will",
    "i've": "I have",
    "i'd": "I would",
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "let's": "let us",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
    flags=re.IGNORECASE,
)


def _expand_contractions(text: str) -> str:
    """Expand common contractions, keeping the original capitalisation."""
    def _swap(match: re.Match) -> str:
        word = match.group(0)
        expanded = _CONTRACTIONS[word.lower()]
        if word[0].isupper():
            expanded = expanded[0].upper() + expanded[1:]
        return expanded

    return _CONTRACTION_RE.sub(_swap, text)


def _strip_punctuation(text: str) -> str:
    """
    Remove punctuation Kokoro would otherwise vocalise or stumble on, but
    keep apostrophes (contractions) and decimal points (3.5).
    """
    text = text.replace("\u2019", "'").replace("\u2018", "'")   # curly quotes
    text = text.replace("%", " percent").replace("&", " and ")
    text = re.sub(r"[-/]", " ", text)                           # well-known -> well known
    text = re.sub(r"[^\w\s'.]", "", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", "", text)              # keep only 3.5-style dots
    return re.sub(r"\s{2,}", " ", text).strip()


def _prepare_for_speech(text: str) -> str:
    text = _strip_punctuation(text)
    if EXPAND_CONTRACTIONS:
        text = _expand_contractions(text)
    return text


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

# ~200 characters per synthesis call is the sweet spot: shorter sounds choppy
# with audible joins, longer delays the first audio.
MAX_CHUNK_CHARS = 200


def _chunk_text(text: str) -> list[str]:
    """
    Split text into natural spoken chunks.

    Short replies are sent as a single chunk so Kokoro doesn't pause at
    every full stop. Longer replies are split on sentence boundaries and
    merged up to MAX_CHUNK_CHARS.
    """
    text = text.strip()
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
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
    """Trim only the leading silence; the tail is left untouched."""
    frame_len = int(sr * 0.01)    # 10 ms frames
    threshold = 10 ** (-top_db / 20.0)

    energy = np.array([
        np.sqrt(np.mean(samples[i:i + frame_len] ** 2))
        for i in range(0, len(samples) - frame_len, frame_len)
    ])

    above = np.where(energy > threshold)[0]
    if len(above) == 0:
        return samples

    return samples[above[0] * frame_len:]