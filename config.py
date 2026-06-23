"""
Jarvis — Cross-platform voice assistant configuration.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent

PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == "Windows"
IS_LINUX   = PLATFORM == "Linux"
IS_MAC     = PLATFORM == "Darwin"


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if IS_MAC and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class STTConfig:
    model_size: str = "base"
    hf_repo: str | None = None
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    min_samples: int = 8000


@dataclass
class LLMConfig:
    model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    hf_repo: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    hf_filename: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    n_gpu_layers: int = 20
    n_threads: int = 10
    context_length: int = 2048
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Forms of address Jarvis will rotate through when speaking to the user.
    # Add, remove, or reorder entries freely.
    user_titles: list = field(default_factory=lambda: [
        "Sir",
        "Mr Madanile",
    ])

    system_prompt: str = (
        "You are Jarvis, an AI assistant. "
        "Speak in plain, natural sentences as if talking aloud. "
        "Never use markdown, bullet points, numbered lists, dashes, or headers. "
        "Never start your reply with a number. "
        "Keep every response to one or two sentences unless the user asks for detail. "
        "Address the user by the title you are given at the start of each instruction."
    )


@dataclass
class TTSConfig:
    voice: str = "bm_george"
    speed: float = 1.05
    sample_rate: int = 24000
    trim_silence: bool = True
    trim_top_db: float = 22.0


@dataclass
class VADConfig:
    sample_rate: int = 16000
    chunk_ms: int = 32
    silence_duration: float = 0.8
    threshold: float = 0.4
    min_speech_duration: float = 0.4


@dataclass
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    beep_path: Path = ROOT / "assets" / "beep.wav"


@dataclass
class Config:
    stt: STTConfig     = field(default_factory=STTConfig)
    llm: LLMConfig     = field(default_factory=LLMConfig)
    tts: TTSConfig     = field(default_factory=TTSConfig)
    vad: VADConfig     = field(default_factory=VADConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    verbose: bool = True
    max_history_turns: int = 6


cfg = Config()