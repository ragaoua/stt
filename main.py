from __future__ import annotations

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
        "--audio-file",
        type=str,
        default=None,
        help="Transcribe an audio file instead of live microphone input",
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
        "--partial-stability",
        type=float,
        default=2.0,
        help=(
            "Seconds at end of window allowed to revise (default: 2.0). "
            "Higher is more stable but lags more."
        ),
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

    if args.audio_file:
        try:
            segments, _info = model.transcribe(
                args.audio_file,
                language=args.language,
                vad_filter=True,
            )
        except Exception as exc:  # pragma: no cover - CLI safety net
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        lines = [segment.text.strip() for segment in segments if segment.text.strip()]
        if lines:
            print(" ".join(lines))
        else:
            print("No speech detected.")
        return

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
        end_sample = total_samples
        start_sample = total_samples - needed
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
        audio = np.concatenate(list(reversed(collected)), axis=0).flatten()
        return audio, start_sample, end_sample

    def _render_transcript_so_far(text: str) -> None:
        # Clear screen + move cursor home (ANSI)
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write("Speak now. Press Ctrl+C to stop and transcribe.\n\n")
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _join_whisper_words(words: list[str]) -> str:
        # faster-whisper word tokens usually include leading spaces.
        return "".join(words).strip()

    try:
        with sd.InputStream(
            samplerate=args.samplerate,
            device=args.device,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            last_partial_at = time.monotonic()
            last_rendered = ""
            committed_words: list[str] = []
            committed_end_s = 0.0
            while True:
                sd.sleep(100)
                if args.partial_interval <= 0:
                    continue
                now = time.monotonic()
                if now - last_partial_at < args.partial_interval:
                    continue
                last_partial_at = now
                tail = tail_audio(args.partial_window)
                if tail is None:
                    continue
                audio, start_sample, end_sample = tail
                segments, _info = model.transcribe(
                    audio,
                    language=args.language,
                    vad_filter=True,
                    word_timestamps=True,
                )

                window_end_s = end_sample / float(args.samplerate)
                committed_until_s = max(
                    0.0, window_end_s - max(0.0, float(args.partial_stability))
                )
                window_start_s = start_sample / float(args.samplerate)

                tol_s = 0.08

                new_words: list[tuple[float, float, str]] = []
                for seg in segments:
                    words = getattr(seg, "words", None)
                    if not words:
                        t = seg.text.strip()
                        if not t:
                            continue
                        abs_s = window_start_s + float(seg.start)
                        abs_e = window_start_s + float(seg.end)
                        new_words.append((abs_s, abs_e, " " + t))
                        continue
                    for w in words:
                        wt = getattr(w, "word", "")
                        if not wt:
                            continue
                        abs_s = window_start_s + float(w.start)
                        abs_e = window_start_s + float(w.end)
                        new_words.append((abs_s, abs_e, wt))

                new_words.sort(key=lambda x: (x[0], x[1]))

                for _s, e, wt in new_words:
                    if e <= committed_end_s + tol_s:
                        continue
                    if e <= committed_until_s - tol_s:
                        committed_words.append(wt)
                        committed_end_s = e

                unstable_words = [
                    wt for _s, e, wt in new_words if e > committed_end_s + tol_s
                ]

                transcript_so_far = _join_whisper_words(
                    committed_words + unstable_words
                )
                if transcript_so_far and transcript_so_far != last_rendered:
                    _render_transcript_so_far(transcript_so_far)
                    last_rendered = transcript_so_far
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
