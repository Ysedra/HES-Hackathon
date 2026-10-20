"""Simple offline ASR test with faster-whisper on CPU.

Transcribes an audio/video file and reports how long processing took
compared with the audio duration (RTF < 1 means faster than real time).

Usage:
    python asr/test_asr.py samples/en.wav
    python asr/test_asr.py samples/fr.wav --model large-v3-turbo --language fr
"""

import argparse
import time

from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="Path to an audio or video file")
    parser.add_argument("--model", default="small", help="Whisper model size or path (default: small)")
    parser.add_argument("--language", default=None, help="Language code, e.g. en or fr (default: auto-detect)")
    parser.add_argument("--compute-type", default="int8", help="CTranslate2 compute type (default: int8)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads, 0 = library default")
    parser.add_argument("--words", action="store_true", help="Print word-level timestamps")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type, cpu_threads=args.threads)
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        args.audio,
        language=args.language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=args.words,
    )
    print(f"Language: {info.language} (p={info.language_probability:.2f})")
    print("-" * 60)
    # segments is a lazy generator: transcription happens while iterating
    for seg in segments:
        print(f"[{seg.start:7.2f} -> {seg.end:7.2f}] {seg.text.strip()}")
        if args.words:
            for w in seg.words:
                print(f"      {w.start:7.2f} {w.end:7.2f}  {w.word.strip()}  ({w.probability:.2f})")
    proc_time = time.perf_counter() - t0

    print("-" * 60)
    print(f"Model: {args.model} ({args.compute_type}, cpu)")
    print(f"Model load: {load_time:.2f} s")
    print(f"Audio duration: {info.duration:.2f} s")
    print(f"Processing: {proc_time:.2f} s  ->  RTF = {proc_time / info.duration:.3f}")


if __name__ == "__main__":
    main()
