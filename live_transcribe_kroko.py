"""
- Live transcription with the Kroko streaming model (sherpa-onnx) of the audio that the computer plays (WASAPI loopback) or of the microphone (SOURCE setting). Works on Windows only.
The audio comes from audio_capture.py in small chunks, and every chunk goes to the model at once.

Differences from live_transcribe.py (Whisper):
- No buffer, no overlap, no queue and no second thread. Kroko is a streaming model: it keeps its own state (the stream) between chunks.
- The encoder runs when the stream has 1.28 s of new audio. One run takes much less time than the next 0.5 s chunk, so one thread is enough.
- With greedy search a word is never changed after it is printed, so we only print the new part of the text.

Endpoint: when the speaker is silent for some time, the model says "end of segment". Then we start a new line and reset the stream.
Rule 1: 2.4 s of silence (also when nothing was said), rule 2: 1.2 s of silence after speech, rule 3: a segment is longer than 20 s.

Run from the root folder:  uv run python live_transcribe_kroko.py
Stop with Ctrl+C.
"""

from pathlib import Path

import sherpa_onnx
from huggingface_hub import snapshot_download

from audio_capture import get_loopback, get_microphone, read_chunk, SAMPLERATE

# Model settings (the same model as in learning/sherpa-onnx-test.ipynb)
KROKO_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
NUM_THREADS = 4
SOURCE = "loopback"   # "loopback" = the sound that goes to the speaker, "microphone" = the sound in the room


def load_recognizer():
    model_dir = Path(snapshot_download(KROKO_REPO))
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / "encoder.onnx"),
        decoder=str(model_dir / "decoder.onnx"),
        joiner=str(model_dir / "joiner.onnx"),
        num_threads=NUM_THREADS,
        sample_rate=SAMPLERATE,
        feature_dim=80,
        model_type="zipformer2",
        decoding_method="greedy_search",
        enable_endpoint_detection=True,
    )


# Give one chunk to the model and print the new part of the text.
# printed is the text of this segment that is already on the screen.
def process_chunk(recognizer, stream, chunk, printed):
    stream.accept_waveform(SAMPLERATE, chunk)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    text = recognizer.get_result(stream)
    if text.startswith(printed):
        print(text[len(printed):], end="", flush=True)
    else:
        print("\n" + text, end="", flush=True)  # should not happen with greedy search
    printed = text

    if recognizer.is_endpoint(stream):
        if printed:
            print()  # end of the segment: new line
        recognizer.reset(stream)
        printed = ""
    return printed


def main():
    print("Loading the model...")
    recognizer = load_recognizer()
    stream = recognizer.create_stream()

    if SOURCE == "microphone":
        source = get_microphone()
    else:
        source = get_loopback()

    print("Recording from:", source)
    print("Press Ctrl+C to stop.")

    printed = ""
    # blocksize = 1 s of audio: the Windows buffer keeps 1 s while the model works.
    # The default is only about 10 ms, so audio was lost ("data discontinuity in recording").
    with source.recorder(samplerate=SAMPLERATE, blocksize=SAMPLERATE) as recorder:
        try:
            while True:
                chunk = read_chunk(recorder)
                printed = process_chunk(recognizer, stream, chunk, printed)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
