"""
tests/test_components.py — Smoke tests for each Jarvis component.

Run with:  pytest tests/ -v
Each test is independent — they can run without a mic or speakers.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

class TestTranscriber:
    def test_import(self):
        from stt import Transcriber
        assert Transcriber

    def test_transcribe_silence(self):
        """Short silence should return an empty string, not raise."""
        from stt import Transcriber
        t = Transcriber()
        silence = np.zeros(4000, dtype=np.float32)   # 0.25 s — below min_samples
        result = t.transcribe(silence)
        assert result == ""

    def test_transcribe_noise(self):
        """White noise should not crash (may return garbage text)."""
        from stt import Transcriber
        t = Transcriber()
        noise = (np.random.randn(24000) * 0.01).astype(np.float32)  # 1.5 s
        result = t.transcribe(noise)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class TestLLMEngine:
    def test_import(self):
        from llm import LLMEngine
        assert LLMEngine

    def test_build_prompt_first_turn(self):
        """First-turn prompt must include the system block."""
        from llm.engine import LLMEngine
        eng = LLMEngine.__new__(LLMEngine)
        from config import cfg
        eng._system = cfg.llm.system_prompt
        eng._history = [{"role": "user", "content": "Hello"}]
        prompt = eng._build_prompt()
        assert "<<SYS>>" in prompt
        assert "Hello" in prompt

    def test_build_prompt_multi_turn(self):
        """Subsequent turns must NOT repeat the system block."""
        from llm.engine import LLMEngine
        eng = LLMEngine.__new__(LLMEngine)
        from config import cfg
        eng._system = cfg.llm.system_prompt
        eng._history = [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi Sir."},
            {"role": "user",      "content": "What time is it?"},
        ]
        prompt = eng._build_prompt()
        # <<SYS>> should appear exactly once
        assert prompt.count("<<SYS>>") == 1

    def test_history_trimming(self, mocker):
        """History must not grow beyond max_history_turns * 2 messages."""
        from llm.engine import LLMEngine
        from config import cfg

        eng = LLMEngine.__new__(LLMEngine)
        eng._system = cfg.llm.system_prompt
        eng._history = []

        # Mock the underlying llama call
        mock_llm = mocker.MagicMock()
        mock_llm.return_value = {"choices": [{"text": "reply"}]}
        eng._llm = mock_llm

        for i in range(cfg.max_history_turns + 3):
            eng.chat(f"message {i}")

        assert len(eng._history) <= cfg.max_history_turns * 2


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

class TestSynthesiser:
    def test_import(self):
        from tts import Synthesiser
        assert Synthesiser

    def test_empty_string(self):
        """Empty input must not crash."""
        from tts import Synthesiser
        s = Synthesiser()
        audio, sr = s.synthesise("")
        assert isinstance(audio, np.ndarray)
        assert sr == 24000

    def test_short_sentence(self):
        """A short sentence must produce non-trivial audio."""
        from tts import Synthesiser
        s = Synthesiser()
        audio, sr = s.synthesise("Hello.")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 100
        assert sr == 24000

    def test_stream_yields_chunks(self):
        """stream() must yield at least one chunk for a non-empty string."""
        from tts import Synthesiser
        s = Synthesiser()
        chunks = list(s.stream("Hello. How are you today?"))
        assert len(chunks) >= 1
        for audio, sr in chunks:
            assert isinstance(audio, np.ndarray)
            assert sr == 24000


# ---------------------------------------------------------------------------
# VAD (unit-level — no microphone)
# ---------------------------------------------------------------------------

class TestVADDetector:
    def test_import(self):
        from stt import VADDetector
        assert VADDetector

    def test_silent_chunk_does_not_emit(self):
        """Feeding silence should never trigger the callback."""
        from stt import VADDetector

        fired = []
        vad = VADDetector(on_speech_end=lambda arr: fired.append(arr))

        silence = (np.zeros(512, dtype=np.float32) * 32767).astype(np.int16).tobytes()
        for _ in range(100):
            vad.process_chunk(silence)

        assert len(fired) == 0

    def test_reset_clears_buffer(self):
        """reset() must clear internal buffers without error."""
        from stt import VADDetector
        vad = VADDetector(on_speech_end=lambda arr: None)
        vad.reset()   # must not raise


# ---------------------------------------------------------------------------
# Audio helpers (no hardware required)
# ---------------------------------------------------------------------------

class TestAudioHelpers:
    def test_trim_silence(self):
        from tts.synthesiser import _trim_silence
        # Pad a short tone with silence
        sr = 24000
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sr * 0.1))).astype(np.float32) * 0.5
        padded = np.concatenate([np.zeros(sr // 2, dtype=np.float32), tone, np.zeros(sr // 2, dtype=np.float32)])
        trimmed = _trim_silence(padded, sr)
        # Trimmed must be shorter than padded
        assert len(trimmed) < len(padded)
        # But must still contain the tone
        assert len(trimmed) > 0

    def test_split_sentences(self):
        from tts.synthesiser import _split_sentences
        text = "Hello. How are you? Fine, thank you!"
        parts = _split_sentences(text)
        assert len(parts) == 3
        assert parts[0] == "Hello."
