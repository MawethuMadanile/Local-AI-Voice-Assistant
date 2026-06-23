"""
stt/transcriber.py — Speech-to-text using faster-whisper.

faster-whisper uses CTranslate2 under the hood:
  - Works on Windows, Linux, macOS
  - Supports CPU (int8), CUDA (float16/int8), and DirectML (Windows)
  - Significantly faster than openai-whisper
  - pip install faster-whisper

For CUDA acceleration on Windows/Linux, install the CUDA runtime:
    pip install faster-whisper
    # and ensure CUDA drivers are installed for your GPU

For CPU-only (all platforms, zero extra setup):
    Uses int8 quantisation by default — fast enough for real-time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from config import cfg


class Transcriber:
    """
    Thin wrapper around faster-whisper.WhisperModel.

    The model is downloaded on first use and cached in
    ~/.cache/huggingface/hub  (or wherever HF_HOME points).
    """

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        device = cfg.stt.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = cfg.stt.compute_type
        # float16 only makes sense on CUDA
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        model_id = cfg.stt.hf_repo or cfg.stt.model_size

        self._model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
        )
        self._language = cfg.stt.language

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 mono array sampled at 16 kHz.

        Returns the plain text string (stripped, no timestamps).
        """
        if len(audio) < cfg.stt.min_samples:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,   # we do our own VAD
            without_timestamps=True,
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text
