# Jarvis — Installation Guide

## Minimum requirements

| | Windows | Linux | macOS |
|---|---|---|---|
| Python | 3.10 – 3.12 | 3.10 – 3.12 | 3.10 – 3.12 |
| RAM | 6 GB | 6 GB | 6 GB |
| Disk | ~5 GB | ~5 GB | ~5 GB |
| GPU (optional) | CUDA 11.8+ or Vulkan | CUDA 11.8+ | Metal (Apple Silicon) |

---

## 1. Clone / extract

```
git clone https://github.com/mawethu/jarvis.git
cd jarvis
```

---

## 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install base dependencies

```
pip install -r requirements.txt
```

This installs CPU-only builds of everything. The assistant will work on any
machine without a GPU — the base.Q4_K_M GGUF LLM runs at ~5–8 tok/s on a
modern CPU, which is fast enough for voice conversation.

---

## 4. GPU acceleration (optional)

### 4a. CUDA (Windows / Linux — NVIDIA)

**LLM (llama-cpp-python with CUDA)**
```
pip uninstall llama-cpp-python -y
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Set `cfg.llm.n_gpu_layers = -1` in `config.py` to offload all layers.

**STT (faster-whisper with CUDA)**
```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
Then set `cfg.stt.device = "cuda"` and `cfg.stt.compute_type = "float16"`.

**TTS (Kokoro — optional GPU)**
```
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

---

### 4b. Vulkan (Windows / Linux — AMD, Intel, NVIDIA without CUDA toolkit)

**LLM only** — llama-cpp-python Vulkan build:
```
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python
```
On Windows, install the Vulkan SDK first: https://vulkan.lunarg.com/sdk/home

---

### 4c. Apple Silicon / Metal (macOS)

llama-cpp-python auto-detects Metal on macOS ARM — no extra steps needed.  
Set `cfg.llm.n_gpu_layers = -1` to use the GPU.

faster-whisper on macOS uses CPU (Metal is not yet supported by CTranslate2).
For comparable speed use `model_size = "base"` or `"small"`.

---

## 5. Download the LLM model

The model is downloaded automatically on first run.

To pre-download manually:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="models/",
)
```

### Alternative models

Any GGUF model works — just update `cfg.llm.model_path`.

| Model | Size | Notes |
|---|---|---|
| Mistral-7B-Instruct-v0.2 Q4_K_M | 4.4 GB | Default, best balance |
| Llama-3.1-8B-Instruct Q4_K_M | 4.7 GB | Slightly better quality |
| Phi-3-mini-4k-instruct Q4 | 2.2 GB | Faster, less capable |
| Gemma-2-2b-it Q4_K_M | 1.6 GB | Fastest, lowest RAM |

---

## 6. Audio device setup

List your audio devices:
```python
from audio import list_devices
list_devices()
```

Set `cfg.audio.input_device` and `cfg.audio.output_device` to the integer
index of your preferred devices if the system default is wrong.

---

## 7. Run

```
python main.py
```

Wait for "Listening…" to appear, then speak.
Press **Ctrl+C** to quit cleanly.

---

## 8. Run tests

```
pip install pytest pytest-mock
pytest tests/ -v
```

---

## Troubleshooting

### "PortAudio not found" (Linux)
```
sudo apt install libportaudio2
```

### "No module named '_sounddevice'" (Windows)
sounddevice ships with PortAudio wheels on Windows — reinstall:
```
pip install --force-reinstall sounddevice
```

### First run is slow
Model download + warm-up takes 30–120 s depending on connection and hardware.
Subsequent starts are fast (models are cached).

### Microphone not detected
Run `python -c "from audio import list_devices; list_devices()"` and set
`cfg.audio.input_device` to the correct index.

### LLM responses are too slow on CPU
Switch to a smaller model: set `cfg.llm.hf_filename = "mistral-7b-instruct-v0.2.Q2_K.gguf"`
or use Phi-3-mini / Gemma-2-2b.
