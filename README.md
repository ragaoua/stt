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

Speak into your microphone. The CLI prints partial transcripts while you talk and outputs a final transcript after you press Ctrl+C.
