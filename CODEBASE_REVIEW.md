# Codebase Review: Jarvis Voice Assistant

## Project Overview

**Jarvis** is a fully offline, cross-platform voice assistant built with modern Python libraries. The project is transitioning from a macOS-only MLX-based solution to a universal solution supporting Windows, Linux, and macOS.

- **Status**: Active development
- **Branch**: `agents/cinthia-profile-update`
- **Primary Goal**: Rename/rebrand from "Jarvis" to "Cinthia" and update profile configuration

---

## Architecture Summary

### Core Pipeline
```
Microphone → VAD Detection → Transcription → LLM Processing → TTS → Speaker
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **VAD** | Silero VAD | Voice Activity Detection (cross-platform, maintained) |
| **STT** | faster-whisper (CTranslate2) | Speech-to-text with CUDA/CPU support |
| **LLM** | llama-cpp-python (GGUF) | Local LLM inference |
| **TTS** | Kokoro ONNX | Text-to-speech (CPU-only, high quality) |
| **Audio** | sounddevice | Cross-platform mic/speaker I/O |

---

## Project Structure

```
jarvis/
├── main.py                  # Entry point - Jarvis class with threading pipeline
├── config.py               # Centralized configuration (all settings)
├── requirements.txt        # Dependencies
├── INSTALL.md             # GPU setup guide
├── README.md              # Project documentation
│
├── audio/
│   ├── __init__.py
│   └── io.py              # MicrophoneStream, play_audio, beep playback
│
├── stt/
│   ├── __init__.py
│   ├── transcriber.py     # faster-whisper wrapper
│   └── vad.py             # Silero VAD wrapper for speech detection
│
├── llm/
│   ├── __init__.py
│   └── engine.py          # llama-cpp-python wrapper + chat history management
│
├── tts/
│   ├── __init__.py
│   └── synthesiser.py     # Kokoro ONNX wrapper + streaming
│
├── tests/
│   └── test_components.py # Component-level smoke tests
│
├── assets/
│   └── beep.wav          # Notification beep sound
│
└── models/               # GGUF files auto-downloaded here
```

---

## Component Analysis

### 1. **Configuration Module** (`config.py`)
- **Dataclasses**: `STTConfig`, `LLMConfig`, `TTSConfig`, `VADConfig`, `AudioConfig`
- **Key Settings**:
  - LLM: Mistral-7B-Instruct-v0.2 GGUF format
  - TTS: British male voice (bm_george)
  - STT: Whisper base model
  - User titles: "Sir", "Mr Madanile" (rotated per turn)
  - System prompt defines assistant behavior
- **Platform Detection**: Auto-detects Windows/Linux/macOS

**Profile-Related Settings**:
```python
user_titles: list = ["Sir", "Mr Madanile"]
system_prompt: str = "You are Jarvis, an AI assistant..."
tts.voice: str = "bm_george"  # Can be: bm_george, bm_lewis, am_michael, etc.
```

### 2. **Main Entry Point** (`main.py`)
- **Class**: `Jarvis` - main orchestrator
- **Threading Model**: 
  - Main thread: listens for KeyboardInterrupt
  - Audio thread (MicrophoneStream): captures audio chunks
  - Pipeline thread: processes VAD → STT → LLM → TTS
- **Pipeline Flow**:
  1. VAD detects speech end → queues audio
  2. STT transcribes to text
  3. LLM generates reply with rotating titles
  4. TTS synthesizes audio
  5. play_audio outputs to speakers
  6. Beep indicates readiness

**Customization Points**:
- Banner printing (lines 206-212)
- User/assistant display labels (lines 201-204)
- System messages and logging

### 3. **STT Module** (`stt/`)
- **Transcriber**: faster-whisper wrapper
  - Model: base (configurable: tiny/base/small/medium/large-v3)
  - Device: auto-detects CUDA/CPU
  - Language: English (configurable)
- **VADDetector**: Silero VAD-based speech detection
  - Detects speech end and queues audio
  - Silence threshold: 0.8s

### 4. **LLM Module** (`llm/engine.py`)
- **Model**: Mistral-7B-Instruct-v0.2 GGUF
- **Chat History**: Maintains multi-turn conversation (trimmed to 6 turns)
- **Title Rotation**: Uses `itertools.cycle()` to rotate user titles
- **Prompt Building**: Injects title dynamically into system prompt
- **Sanitization**: Removes markdown, bullet points, headers from responses
- **Key Methods**:
  - `chat()`: Takes user text, returns assistant reply
  - `_build_prompt()`: Constructs Mistral instruction format
  - `clear_history()`: Resets conversation

### 5. **TTS Module** (`tts/synthesiser.py`)
- **Voice Engine**: Kokoro ONNX v1.0
- **Voice Options**:
  - British: bm_george (default), bm_lewis
  - American: am_michael, am_adam, af_sarah, bf_emma
- **Streaming**: Sentence-by-sentence for low latency
- **Audio Processing**:
  - Trimming: Removes leading silence only (preserves end)
  - Padding: 20ms (mid-sentence) or 120ms (final) to prevent clipping
  - Chunk size: ~200 characters for optimal pacing

### 6. **Audio Module** (`audio/io.py`)
- **Cross-platform I/O**: Single library (sounddevice)
- **Functions**:
  - `MicrophoneStream`: Async microphone capture
  - `play_audio()`: Play float32 numpy arrays
  - `play_beep()`: Non-blocking notification sound
  - `stop_audio()`: Stop playback
- **Audio Format**: float32, configurable sample rates

### 7. **Testing** (`tests/test_components.py`)
- **Test Coverage**: STT, LLM, TTS, VAD, Audio helpers
- **Test Types**: Import tests, unit tests, mock integration tests
- **Run Command**: `pytest tests/ -v`

---

## Key Configuration Points for "Cinthia" Profile Update

To transition from Jarvis → Cinthia, these files need updates:

### Files Requiring Changes

1. **`config.py`**
   - [ ] User titles (currently ["Sir", "Mr Madanile"])
   - [ ] System prompt (currently references "Jarvis")
   - [ ] TTS voice selection (currently "bm_george")
   - [ ] Any profile-specific settings

2. **`main.py`**
   - [ ] Banner/display text (currently "JARVIS")
   - [ ] System messages and labels
   - [ ] Any Jarvis-specific branding

3. **`README.md` & `INSTALL.md`**
   - [ ] Project name references
   - [ ] Configuration examples
   - [ ] Quick start guide

4. **Other files**
   - [ ] Comments/docstrings mentioning Jarvis
   - [ ] Configuration examples in docstrings

---

## Current Issues & Notes

### Potential Areas for Improvement

1. **Profile Management**
   - Currently no profile switching mechanism
   - All settings are hardcoded in `config.py`
   - Could benefit from profile files or environment-based config

2. **Error Handling**
   - Model download fails with RuntimeError but could be more graceful
   - Transcription errors logged but not user-visible

3. **History Management**
   - Chat history is in-memory only
   - Could persist for longer-term context

4. **Testing Coverage**
   - Tests are mostly unit/integration smoke tests
   - Missing end-to-end pipeline tests
   - Missing performance benchmarks

---

## Next Steps (Cinthia Profile Update)

1. **Update Configuration** - Rename Jarvis to Cinthia in config.py
2. **Update Branding** - Change banner, labels, system prompt in main.py
3. **Update Documentation** - Reflect new name in README, comments
4. **Profile-Specific Settings** - Consider what makes Cinthia unique
5. **Testing** - Verify all components work with new configuration
6. **Commit** - Create pull request with changes

---

## Performance Notes

- **Startup Time**: ~15-30s (model loading)
- **Latency**: ~200-500ms STT + 100-300ms LLM + 50-100ms TTS per response
- **CPU Usage**: 
  - Idle: 0-2%
  - Processing: 20-40% (can reach 100% during LLM inference on older systems)
- **Memory**: ~2-3GB with all models loaded

---

## Security Considerations

- All processing is local (no cloud calls)
- No credentials or API keys required
- Microphone input is only captured on explicit voice activity
- Audio is not persisted to disk by default

---

Generated: 2026-08-31 | Branch: agents/cinthia-profile-update
