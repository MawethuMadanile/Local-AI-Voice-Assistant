"""
assets/generate_beep.py — Generate beep.wav if it doesn't already exist.

Run once:  python assets/generate_beep.py
Or it is called automatically from audio/io.py if the file is missing.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_beep(
    path: Path = Path(__file__).parent / "beep.wav",
    freq: float = 880.0,
    duration: float = 0.12,
    volume: float = 0.35,
    sample_rate: int = 44100,
) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    # Smooth fade in/out to prevent clicks
    fade_samples = int(sample_rate * 0.008)
    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    sf.write(str(path), wave, sample_rate)
    print(f"Wrote {path}")


if __name__ == "__main__":
    generate_beep()
