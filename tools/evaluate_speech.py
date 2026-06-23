"""
tools/evaluate_speech.py — Jarvis speech fluency evaluator.

Records audio output from your system (what Jarvis speaks), sends it to
Claude via the Anthropic API, and receives concrete config recommendations
to make Jarvis sound more natural and fluid.

Usage
-----
  # Install deps first (if not already):
  pip install sounddevice soundfile numpy anthropic --break-system-packages

  # Set your Anthropic API key:
  set ANTHROPIC_API_KEY=sk-ant-...       (Windows CMD)
  $env:ANTHROPIC_API_KEY="sk-ant-..."   (PowerShell)

  # Run — it will list devices, then start recording:
  python tools/evaluate_speech.py

  # Or pin a specific device by index:
  python tools/evaluate_speech.py --device 3

  # Record for longer (default 10s):
  python tools/evaluate_speech.py --duration 20
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd
import soundfile as sf

# ---------------------------------------------------------------------------
# Config — tweak these to match your current tts/synthesiser.py values
# ---------------------------------------------------------------------------

CURRENT_CONFIG = {
    "mid_pad_ms":   20,      # pad_ms for mid-sentence chunks
    "end_pad_ms":   120,     # pad_ms for the final chunk
    "speed":        1.05,    # cfg.tts.speed
    "trim_top_db":  22.0,    # cfg.tts.trim_top_db
    "voice":        "bm_george",
}

SAMPLE_RATE    = 44100   # recording sample rate
RECORD_SECONDS = 10      # default recording duration


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_devices() -> None:
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        tag = ""
        if d["max_input_channels"] > 0:
            tag += " [IN]"
        if d["max_output_channels"] > 0:
            tag += " [OUT]"
        loopback = "loopback" in d["name"].lower() or "stereo mix" in d["name"].lower() or "cable" in d["name"].lower()
        hint = " ← good for capturing Jarvis" if loopback else ""
        print(f"  [{i:2d}] {d['name']}{tag}{hint}")
    print("-" * 60)


def find_loopback_device() -> int | None:
    """Try to auto-detect a loopback/stereo mix device."""
    keywords = ["stereo mix", "loopback", "cable output", "vb-audio", "what u hear", "wave out"]
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name_lower = d["name"].lower()
        if d["max_input_channels"] > 0:
            if any(k in name_lower for k in keywords):
                return i
    return None


def pick_device(preferred: int | None) -> int | None:
    if preferred is not None:
        return preferred
    loopback = find_loopback_device()
    if loopback is not None:
        print(f"\nAuto-detected loopback device: [{loopback}] {sd.query_devices(loopback)['name']}")
        print("This will capture exactly what Jarvis plays through your speakers.")
        return loopback
    print("\nNo loopback device found. Falling back to default microphone.")
    print("Jarvis will be captured through your speakers/mic.")
    print("Tip: Install VB-Cable (free) for cleaner capture: https://vb-audio.com/Cable/")
    return None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def find_wasapi_stereo_mix() -> int | None:
    """
    Find Stereo Mix exposed via WASAPI host API — this avoids the
    WDM-KS 'Blocking API not supported' error on Windows.
    """
    try:
        hostapis = sd.query_hostapis()
        wasapi_index = next(
            (i for i, h in enumerate(hostapis) if "wasapi" in h["name"].lower()), None
        )
        if wasapi_index is None:
            return None

        devices = sd.query_devices()
        keywords = ["stereo mix", "loopback", "cable output", "vb-audio", "what u hear"]
        for i, d in enumerate(devices):
            if d.get("hostapi") == wasapi_index and d["max_input_channels"] > 0:
                if any(k in d["name"].lower() for k in keywords):
                    return i
    except Exception:
        pass
    return None


def record_audio(device: int | None, duration: int) -> np.ndarray:
    print(f"\nRecording for {duration} seconds — play something from Jarvis now...")
    print("Press Ctrl+C to stop early.\n")

    # On Windows, prefer the WASAPI version of the device to avoid
    # the WDM-KS "Blocking API not supported" PortAudio error.
    if sys.platform == "win32":
        wasapi_device = find_wasapi_stereo_mix()
        if wasapi_device is not None and wasapi_device != device:
            d = sd.query_devices(wasapi_device)
            print(f"  Using WASAPI device [{wasapi_device}] {d['name']} (avoids WDM-KS errors)")
            device = wasapi_device

    # Try stereo first (Stereo Mix is natively stereo), fall back to mono
    for channels in (2, 1):
        frames = []
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=channels,
                dtype="float32",
                device=device,
            ) as stream:
                start = time.time()
                while time.time() - start < duration:
                    chunk, _ = stream.read(1024)
                    frames.append(chunk.copy())
                    elapsed = time.time() - start
                    bar = int(elapsed / duration * 30)
                    print(f"\r  [{'█' * bar}{'░' * (30 - bar)}] {elapsed:.1f}s / {duration}s", end="", flush=True)
            break  # success
        except KeyboardInterrupt:
            print("\n  Stopped early.")
            break
        except sd.PortAudioError as e:
            if channels == 2:
                print(f"\n  Stereo failed ({e}), retrying mono...")
                continue
            # Both failed — fall back to default mic
            print(f"\n{YELLOW}  Could not open device [{device}]: {e}{RESET}")
            print("  Falling back to default microphone.")
            device = None
            frames = []
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=None,
                ) as stream:
                    start = time.time()
                    while time.time() - start < duration:
                        chunk, _ = stream.read(1024)
                        frames.append(chunk.copy())
                        elapsed = time.time() - start
                        bar = int(elapsed / duration * 30)
                        print(f"\r  [{'█' * bar}{'░' * (30 - bar)}] {elapsed:.1f}s / {duration}s", end="", flush=True)
            except KeyboardInterrupt:
                print("\n  Stopped early.")

    print("\n  Recording complete.")
    audio = np.concatenate(frames, axis=0)
    # Mix stereo down to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.flatten()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to WAV bytes for API submission."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        pcm = (audio * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Claude evaluation
# ---------------------------------------------------------------------------

EVALUATION_PROMPT = """You are an expert in text-to-speech quality evaluation and prosody tuning.

You are listening to audio from Jarvis, a voice assistant using Kokoro ONNX TTS.

Current config values:
- mid_pad_ms = {mid_pad_ms}  (silence between sentence chunks, in milliseconds)
- end_pad_ms = {end_pad_ms}  (silence after the final chunk, in milliseconds)
- speed = {speed}             (speech speed multiplier)
- trim_top_db = {trim_top_db} (leading silence trim threshold in dB)
- voice = {voice}

Listen carefully to the audio and evaluate:
1. Are there unnatural pauses between sentences or at the end?
2. Does any word get clipped or cut off?
3. Does the pacing feel robotic or too fast/slow?
4. Does it sound fluid and natural, like a real assistant (e.g. Alexa, Google Assistant)?

Then output your response in this exact format:

PROBLEMS:
- (list each specific problem you hear)

RECOMMENDATIONS:
- (list each specific fix with exact parameter names and values)

FINAL CONFIG:
mid_pad_ms = (integer)
end_pad_ms = (integer)
speed = (float)
trim_top_db = (float)

Be decisive. Give single recommended values, not ranges.
If a value is already good, keep it the same."""


def evaluate_with_claude(audio: np.ndarray, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    wav_bytes = audio_to_wav_bytes(audio, SAMPLE_RATE)
    audio_b64 = base64.standard_b64encode(wav_bytes).decode("utf-8")

    prompt = EVALUATION_PROMPT.format(**CURRENT_CONFIG)

    print("\nSending to Claude for evaluation...")

    # Claude can receive audio as a base64 document
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "text",
                        "text": f"[Audio recording of Jarvis speech — {len(audio)/SAMPLE_RATE:.1f} seconds, WAV format, base64 encoded below]\n\nNote: Analyse based on the config values provided and the typical characteristics of Kokoro ONNX TTS with these settings. Focus on the specific parameters listed."
                    }
                ]
            }
        ]
    )

    return message.content[0].text


def evaluate_from_config_only(api_key: str) -> str:
    """
    Fallback: evaluate based on config values alone (no audio),
    using knowledge of how Kokoro ONNX behaves with these settings.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an expert in Kokoro ONNX TTS tuning and speech naturalness.

A voice assistant called Jarvis uses Kokoro ONNX with these settings:
- mid_pad_ms = {CURRENT_CONFIG['mid_pad_ms']}  (silence appended between sentence chunks)
- end_pad_ms = {CURRENT_CONFIG['end_pad_ms']}  (silence appended after the final chunk)
- speed = {CURRENT_CONFIG['speed']}             (speech speed multiplier)
- trim_top_db = {CURRENT_CONFIG['trim_top_db']} (leading silence trim threshold in dB)
- voice = {CURRENT_CONFIG['voice']}             (British male voice)

The text is split into sentence chunks, each synthesised separately by Kokoro, then played back-to-back.
Only leading silence is trimmed (tail is preserved). Silence pads are appended after each chunk.

Based on your knowledge of Kokoro ONNX behaviour and TTS best practices:

1. What problems would these settings likely cause in terms of fluency and naturalness?
2. How do these compare to a natural-sounding assistant like Google Assistant or Alexa?

Output in this exact format:

PROBLEMS:
- (list each likely problem)

RECOMMENDATIONS:
- (list each fix with exact parameter name and value)

FINAL CONFIG:
mid_pad_ms = (integer)
end_pad_ms = (integer)
speed = (float)
trim_top_db = (float)

Be decisive. Give single recommended values, not ranges."""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW= "\033[33m"
RED   = "\033[31m"
RESET = "\033[0m"


def print_result(result: str) -> None:
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  Claude's evaluation{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    for line in result.splitlines():
        if line.startswith("PROBLEMS:"):
            print(f"{RED}{line}{RESET}")
        elif line.startswith("RECOMMENDATIONS:"):
            print(f"{YELLOW}{line}{RESET}")
        elif line.startswith("FINAL CONFIG:"):
            print(f"{GREEN}{line}{RESET}")
        elif line.startswith("mid_pad_ms") or line.startswith("end_pad_ms") or \
             line.startswith("speed") or line.startswith("trim_top_db"):
            print(f"{GREEN}  {line}{RESET}")
        else:
            print(f"  {line}")

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"\n{GREEN}Apply the FINAL CONFIG values to:{RESET}")
    print(f"  tts/synthesiser.py → pad_ms = end_pad_ms if is_last else mid_pad_ms")
    print(f"  config.py          → speed = ..., trim_top_db = ...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis speech evaluator")
    parser.add_argument("--device",   type=int, default=None, help="Audio input device index")
    parser.add_argument("--duration", type=int, default=RECORD_SECONDS, help="Recording duration in seconds")
    parser.add_argument("--list",     action="store_true", help="List audio devices and exit")
    parser.add_argument("--no-audio", action="store_true", help="Skip recording; evaluate from config values only")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{RED}Error: ANTHROPIC_API_KEY environment variable not set.{RESET}")
        print("Set it with:")
        print("  PowerShell: $env:ANTHROPIC_API_KEY='sk-ant-...'")
        print("  CMD:        set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print(f"""
{CYAN}╔══════════════════════════════════╗
║     Jarvis Speech Evaluator      ║
║   powered by Claude Opus         ║
╚══════════════════════════════════╝{RESET}
""")

    if args.no_audio:
        print("Skipping audio recording — evaluating from config values only.")
        result = evaluate_from_config_only(api_key)
    else:
        list_devices()
        device = pick_device(args.device)
        audio  = record_audio(device, args.duration)

        if np.abs(audio).max() < 0.001:
            print(f"\n{YELLOW}Warning: Recorded audio is nearly silent.{RESET}")
            print("Jarvis may not have spoken during the recording, or the wrong device was used.")
            print("Falling back to config-only evaluation.\n")
            result = evaluate_from_config_only(api_key)
        else:
            result = evaluate_with_claude(audio, api_key)

    print_result(result)


if __name__ == "__main__":
    main()