import argparse
import datetime as _dt
import os
import sys
import time
import wave

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
        type=str,
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
        type=str,
        help="Language code hint (e.g. en, fr). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        type=str,
        help="Compute type (default: auto, examples: int8, float16)",
    )
    parser.add_argument(
        "--partial-interval",
        type=float,
        default=4.0,
        help="Seconds between partial transcriptions (default: 4.0)",
    )
    parser.add_argument(
        "--partial-window",
        type=float,
        default=8.0,
        help="Seconds of recent audio for partial transcription (default: 8.0)",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Disable saving captured audio to a .wav file (saving is on by default)",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help=(
            "Path to write captured audio (.wav). Defaults to ./recordings/<timestamp>.wav"
        ),
    )
    args = parser.parse_args()

    def save_wav_mono16(path: str, audio_f32: np.ndarray, samplerate: int) -> None:
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_i16.tobytes())

    try:
        model = WhisperModel(
            args.model,
            compute_type=args.compute_type,
            download_root="./whisper_models/",
        )
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        sys.exit(1)

    audio_chunks = []
    chunk_sizes = []
    total_samples = 0

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        nonlocal total_samples
        chunk = indata.copy()
        audio_chunks.append(chunk)
        chunk_sizes.append(len(chunk))
        total_samples += len(chunk)

    print("Speak now. Press Ctrl+C to stop and transcribe.")

    def tail_audio(seconds):
        target = int(args.samplerate * seconds)
        if total_samples == 0:
            return None
        needed = min(target, total_samples)
        collected = []
        remaining = needed
        for chunk, size in zip(reversed(audio_chunks), reversed(chunk_sizes)):
            if remaining <= 0:
                break
            if size <= remaining:
                collected.append(chunk)
                remaining -= size
            else:
                collected.append(chunk[-remaining:])
                remaining = 0
        if not collected:
            return None
        return np.concatenate(list(reversed(collected)), axis=0).flatten()

    try:
        with sd.InputStream(
            samplerate=args.samplerate,
            device=args.device,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            last_partial_at = time.monotonic()
            last_partial_text = ""
            while True:
                sd.sleep(100)
                if args.partial_interval <= 0:
                    continue
                now = time.monotonic()
                if now - last_partial_at < args.partial_interval:
                    continue
                last_partial_at = now
                audio = tail_audio(args.partial_window)
                if audio is None:
                    continue
                segments, _info = model.transcribe(
                    audio,
                    language=args.language,
                    vad_filter=True,
                )
                partial_text = " ".join(
                    segment.text.strip() for segment in segments if segment.text.strip()
                )
                if partial_text and partial_text != last_partial_text:
                    print(f"Partial: {partial_text}")
                    last_partial_text = partial_text
    except KeyboardInterrupt:
        print("\nTranscribing...")
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not audio_chunks:
        print("No audio captured.")
        return

    audio = np.concatenate(audio_chunks, axis=0).flatten()

    if not args.no_save_audio:
        audio_path = args.audio_path
        if not audio_path:
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join("recordings", f"audio_{ts}.wav")
        try:
            save_wav_mono16(audio_path, audio, args.samplerate)
            print(f"Saved audio: {audio_path}")
        except Exception as exc:
            print(f"Warning: failed to save audio: {exc}", file=sys.stderr)

    segments, _info = model.transcribe(audio, language=args.language, vad_filter=True)
    lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    if lines:
        print(" ".join(lines))
    else:
        print("No speech detected.")


if __name__ == "__main__":
    main()
