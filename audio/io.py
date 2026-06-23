"""
audio/io.py — Cross-platform audio input/output using sounddevice only.

No PyAudio. No playsound. One library for everything.
"""

import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import cfg


# ---------------------------------------------------------------------------
# Beep
# ---------------------------------------------------------------------------

def play_beep() -> None:
    """Play the notification beep (non-blocking)."""
    path = cfg.audio.beep_path
    if not path.exists():
        # Synthesise a simple 440 Hz beep if the file is missing
        _play_tone(freq=440, duration=0.12, volume=0.4)
        return
    data, sr = sf.read(str(path), dtype="float32")
    sd.play(data, samplerate=sr, device=cfg.audio.output_device, blocking=False)


def _play_tone(freq: float = 440, duration: float = 0.15, volume: float = 0.3) -> None:
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Fade in/out to avoid clicks
    fade = int(sr * 0.01)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    sd.play(wave, samplerate=sr, device=cfg.audio.output_device, blocking=True)


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------

def play_audio(data: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    """
    Play a float32 numpy array.

    Parameters
    ----------
    data        : float32 mono or stereo array
    sample_rate : sample rate of the audio
    blocking    : if True, block until playback finishes
    """
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Ensure loudness is reasonable (peak-normalise if clipping)
    peak = np.abs(data).max()
    if peak > 1.0:
        data = data / peak

    sd.play(data, samplerate=sample_rate, device=cfg.audio.output_device, blocking=blocking)


def stop_audio() -> None:
    sd.stop()


# ---------------------------------------------------------------------------
# Microphone capture
# ---------------------------------------------------------------------------

class MicrophoneStream:
    """
    Continuous microphone capture into a queue of raw int16 chunks.

    Each chunk is exactly `chunk_samples` samples at `sample_rate` Hz.
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_ms: int,
        on_chunk: Callable[[bytes], None],
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.on_chunk = on_chunk
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # indata shape: (frames, channels)  dtype: int16
        mono = indata[:, 0].tobytes()
        self.on_chunk(mono)

    def start(self) -> None:
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_samples,
            device=cfg.audio.input_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def list_devices() -> None:
    """Pretty-print all available audio devices — useful for config."""
    print(sd.query_devices())
