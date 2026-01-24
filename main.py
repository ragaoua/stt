import argparse
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(
        description="Offline speech-to-text CLI using Faster Whisper"
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name or path (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device ID (see sounddevice query if needed)",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Input sample rate (default: 16000)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code hint (e.g. en, fr). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Compute type (default: auto, examples: int8, float16)",
    )
    args = parser.parse_args()

    try:
        model = WhisperModel(args.model, compute_type=args.compute_type)
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        sys.exit(1)

    audio_chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_chunks.append(indata.copy())

    print("Speak now. Press Ctrl+C to stop and transcribe.")

    try:
        with sd.InputStream(
            samplerate=args.samplerate,
            device=args.device,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nTranscribing...")
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not audio_chunks:
        print("No audio captured.")
        return

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    segments, _info = model.transcribe(audio, language=args.language, vad_filter=True)
    lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    if lines:
        print(" ".join(lines))
    else:
        print("No speech detected.")


if __name__ == "__main__":
    main()
