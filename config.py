"""
ODIN — Offline Distributed Intelligence Network: configuration.
"""

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
    voice: str = "am_michael"
    speed: float = 1.05


@dataclass
class LLMConfig:
    # Phi-3 mini 4k: 3.8B params, ~2.3 GB, the viable size for this hardware
    model_path: str = "models/Phi-3-mini-4k-instruct-q4.gguf"
    hf_repo: str = "microsoft/Phi-3-mini-4k-instruct-gguf"
    hf_filename: str = "Phi-3-mini-4k-instruct-q4.gguf"

    n_gpu_layers: int = 0
    n_threads: int = 8
    context_length: int = 4096
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Forms of address, cycled one per reply. Edit or shorten to taste.
    user_titles: list[str] = field(default_factory=lambda: ["Sir", "Mr Madanile"])

    system_prompt: str = (
        "You are ODIN, an AI assistant. "
        "Your name stands for Offline Distributed Intelligence Network. "
        "Speak in plain, natural spoken sentences only. "
        "Never use markdown, code blocks, backticks, bullet points, "
        "numbered lists, dashes, or headers. "
        "Never start your reply with a number. "
        "Never ask follow-up questions. "
        "Answer directly and concisely in one or two sentences. "
        "If you cannot do something, say so in one sentence."
    )


@dataclass
class TTSConfig:
    # af_bella: American female, expressive
    # Male options: bm_george (British), am_michael (American)
    voice: str = "af_bella"
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