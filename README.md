# Jarvis — Cross-platform Offline Voice Assistant

A fully offline voice assistant that runs on **Windows, Linux, and macOS**
using modern, actively-maintained libraries.

```
You speak → VAD detects utterance → Whisper transcribes → Mistral replies → Kokoro speaks
```

Everything runs locally. No cloud, no API keys, no internet after first download.

---

## Platform support

| Feature | Windows | Linux | macOS (Intel) | macOS (Apple Silicon) |
|---|---|---|---|---|
| CPU inference | ✅ | ✅ | ✅ | ✅ |
| CUDA GPU | ✅ | ✅ | ❌ | ❌ |
| Vulkan GPU | ✅ | ✅ | ❌ | ❌ |
| Metal GPU | ❌ | ❌ | ❌ | ✅ (LLM only) |

---

## Component stack

| Component | Library | Why |
|---|---|---|
| **VAD** | [Silero VAD](https://github.com/snakers4/silero-vad) | Cross-platform, maintained, 94%+ accuracy |
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 — fastest Whisper on CPU+CUDA |
| **LLM** | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | GGUF, CPU+CUDA+Vulkan+Metal |
| **TTS** | [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) | High quality, Apache 2.0, no GPU required |
| **Audio** | [sounddevice](https://python-sounddevice.readthedocs.io) | Single library for both mic and speaker |

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

See [INSTALL.md](INSTALL.md) for GPU acceleration and troubleshooting.

---

## Configuration

All settings live in `config.py`. Key options:

```python
# Change LLM model
cfg.llm.hf_repo      = "TheBloke/Phi-3-mini-4k-instruct-GGUF"
cfg.llm.hf_filename  = "Phi-3-mini-4k-instruct-fp16.gguf"

# Enable GPU (CUDA/Vulkan)
cfg.llm.n_gpu_layers = -1          # offload all layers

# Change voice
cfg.tts.voice = "bm_george"        # British male

# Change Whisper model size (tiny/base/small/medium/large-v3)
cfg.stt.model_size = "small"
```

---

## Project structure

```
jarvis/
├── main.py              # Entry point and pipeline orchestration
├── config.py            # All configurable settings
├── requirements.txt
├── INSTALL.md
│
├── audio/
│   ├── __init__.py
│   └── io.py            # MicrophoneStream, play_audio, play_beep
│
├── stt/
│   ├── __init__.py
│   ├── vad.py           # Silero VAD wrapper
│   └── transcriber.py   # faster-whisper wrapper
│
├── llm/
│   ├── __init__.py
│   └── engine.py        # llama-cpp-python wrapper + chat history
│
├── tts/
│   ├── __init__.py
│   └── synthesiser.py   # Kokoro ONNX wrapper
│
├── assets/
│   └── beep.wav
│
├── models/              # GGUF files stored here (auto-downloaded)
│
└── tests/
    └── test_components.py
```

---

## What changed from jarvis-mlx

| Aspect | Old (jarvis-mlx) | New (jarvis) |
|---|---|---|
| Platform | macOS + Apple Silicon only | Windows · Linux · macOS |
| STT | Custom MLX Whisper fork | faster-whisper (CTranslate2) |
| LLM | mlx-lm (MLX only) | llama-cpp-python (GGUF, universal) |
| TTS | Custom MeloTTS (PyTorch) | Kokoro ONNX (CPU, no GPU needed) |
| VAD | webrtcvad (unmaintained 2017) | Silero VAD (actively maintained) |
| Audio | PyAudio + sounddevice + playsound | sounddevice only |
| Chat template | Phi-3 tags on Llama-3 (bug) | Mistral instruct (correct) |
| CPU usage | Busy-poll loop (100% core) | Blocking queue with timeout |
| History | System prompt repeated every turn | Correct first-turn injection |
