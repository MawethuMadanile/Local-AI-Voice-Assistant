# ODIN — Offline Distributed Intelligence Network

A fully offline voice assistant for **Windows, Linux and macOS**, built to run on ordinary low-spec laptops with no GPU.

```
You speak → VAD detects utterance → Whisper transcribes → Phi-3 replies → Kokoro speaks
```

Everything runs locally. No cloud, no API keys, no internet after the first model download.

---

## Why it exists

ODIN is an exercise in making a conversational AI work inside tight hardware limits. It is developed on an Intel i5-1235U laptop with about 8 GB of RAM and no dedicated GPU, and every model choice follows from that: small quantised models, CPU inference and a single audio library.

---

## Platform support

| Feature | Windows | Linux | macOS (Intel) | macOS (Apple Silicon) |
|---|---|---|---|---|
| CPU inference | ✅ | ✅ | ✅ | ✅ |
| CUDA GPU | ✅ | ✅ | ❌ | ❌ |
| Vulkan GPU | ✅ | ✅ | ❌ | ❌ |
| Metal GPU | ❌ | ❌ | ❌ | ✅ (LLM only) |

Developed and tested on Windows with Python 3.14. Linux and macOS are supported by the libraries used but are less tested.

---

## Component stack

| Component | Library | Why |
|---|---|---|
| **VAD** | [Silero VAD](https://github.com/snakers4/silero-vad) | Cross-platform, maintained, accurate |
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2, the fastest Whisper on CPU |
| **LLM** | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | GGUF models on CPU, CUDA, Vulkan or Metal |
| **TTS** | [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) | High quality, Apache 2.0, no GPU needed |
| **Audio** | [sounddevice](https://python-sounddevice.readthedocs.io) | One library for both microphone and speaker |

The default language model is **Phi-3 mini 4k (Q4, about 2.3 GB)**. It is the largest model that responds at a usable speed on this hardware.

---

## Quick start

```bash
pip install -r requirements.txt
```

Download the two Kokoro voice files into the project root (about 300 MB):

```bash
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Then run it:

```bash
python main.py
```

On the first run the Phi-3 model downloads automatically into `models/`. After that no internet is needed. Press Ctrl+C to quit.

---

## Configuration

All settings live in `config.py`. The most useful ones:

```python
# LLMConfig
model_path: str = "models/Phi-3-mini-4k-instruct-q4.gguf"
n_threads: int = 8            # CPU threads for the LLM
n_gpu_layers: int = 0         # -1 offloads every layer to a CUDA/Vulkan GPU
max_tokens: int = 256         # maximum length of a reply
user_titles = ["Sir", ...]    # forms of address, cycled each reply

# TTSConfig
voice: str = "am_michael"     # also: bm_george, bf_emma, af_bella, af_sarah ...
speed: float = 1.05

# STTConfig
model_size: str = "base"      # tiny / base / small / medium / large-v3

# Config
max_history_turns: int = 6    # conversation turns kept in context
```

---

## Memory

Recent conversation is saved to `data/history.json` after every reply and reloaded on start-up, so ODIN remembers across restarts. Only the last `max_history_turns` exchanges are kept, which keeps replies fast inside Phi-3's 4096-token window. Delete the file to start fresh. `data/` is git-ignored, since it contains your conversations.

---

## Project structure

```
Local-AI-Voice-Assistant/
├── main.py              # Entry point and pipeline orchestration
├── config.py            # All configurable settings
├── requirements.txt
│
├── audio/
│   └── io.py            # MicrophoneStream, play_audio, play_beep
│
├── stt/
│   ├── vad.py           # Silero VAD wrapper
│   └── transcriber.py   # faster-whisper wrapper
│
├── llm/
│   └── engine.py        # llama-cpp-python wrapper, Phi-3 prompt format, memory
│
├── tts/
│   └── synthesiser.py   # Kokoro ONNX wrapper, text clean-up and chunking
│
├── assets/
│   └── beep.wav
│
├── models/              # GGUF files (auto-downloaded, git-ignored)
└── data/                # Saved conversation history (git-ignored)
```

`kokoro-v1.0.onnx` and `voices-v1.0.bin` sit in the project root and are also git-ignored.

---

## Known limitations

- **Response time.** A reply on a CPU-only laptop takes a few seconds. Phi-3 mini is also much less capable than current cloud models.
- **Pauses between long replies.** Long answers are spoken in chunks, and there can be a short gap between them.
- **No wake word.** ODIN listens continuously and transcribes anything it hears.
- **No interruption.** It cannot be interrupted while it is speaking.

## Roadmap

- Stream LLM output into the speech engine sentence by sentence
- Wake word detection ("Hey ODIN") with openWakeWord
- Barge-in, so you can interrupt a reply
- Filter for Whisper's false transcriptions on silence or noise
- Error recovery for model failures
- Optional cloud backend, with the local model as an offline fallback

---

## History

ODIN began as Jarvis, then Nova, itself a rewrite of an earlier macOS-only project (jarvis-mlx).

| Aspect | Original (jarvis-mlx) | ODIN |
|---|---|---|
| Platform | macOS + Apple Silicon only | Windows, Linux, macOS |
| STT | Custom MLX Whisper fork | faster-whisper (CTranslate2) |
| LLM | mlx-lm (MLX only) | llama-cpp-python (GGUF, universal) |
| TTS | Custom MeloTTS (PyTorch) | Kokoro ONNX (CPU, no GPU needed) |
| VAD | webrtcvad (unmaintained) | Silero VAD (maintained) |
| Audio | PyAudio + sounddevice + playsound | sounddevice only |
| Prompt format | Phi-3 tags on Llama-3 (bug) | Phi-3 chat template |
| CPU usage | Busy-poll loop (100% of a core) | Blocking queue with timeout |
| History | System prompt repeated every turn | System prompt added once per call, history saved to disk |