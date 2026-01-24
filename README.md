# Speech-to-Text CLI (Faster Whisper)

Minimal offline speech-to-text CLI built with Python, Faster Whisper, and SoundDevice.

## Setup

1. Install dependencies with uv:

```bash
uv sync
```

## Usage

```bash
uv run python main.py --model base
```

Speak into your microphone. Press Ctrl+C to stop recording and transcribe the captured audio.
