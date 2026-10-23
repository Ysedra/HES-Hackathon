"""
- Live transcription with the Kroko streaming model (sherpa-onnx) of the audio that the computer plays (WASAPI loopback) or of the microphone (SOURCE setting). Works on Windows only.
The audio comes from audio_capture.py in small chunks, and every chunk goes to the model at once.

Differences from live_transcribe.py (Whisper):
- No buffer, no overlap, no queue and no second thread. Kroko is a streaming model: it keeps its own state (the stream) between chunks.
- The encoder runs when the stream has 1.28 s of new audio. One run takes much less time than the next 0.5 s chunk, so one thread is enough.
- With greedy search a token is never changed after the model gives it.

Sentences: Kroko writes punctuation. When a token ends with . ? or ! the sentence is finished, because greedy search does not change it later.
So we do not wait for a pause: every finished sentence is printed at once with its time, and the unfinished part is shown on the last line.
Every sentence is made into start_ms, end_ms and text (make_segment), ready for the transcript_segments table (one sentence = one row).
The times are from the start of the stream (when this script started). Writing to the database is not done here yet.

Endpoint: when the speaker is silent for some time, the model says "end of segment". The text without end punctuation is also a sentence then,
and the stream is reset. Rule 1: 2.4 s of silence (also when nothing was said), rule 2: 1.2 s of silence after speech,
rule 3: the segment is longer than RULE3_SECONDS. With the default 20 s, rule 3 cut words in half ("epis" / "ode"),
so it is much longer here: we do not need the endpoint to split the sentences.

Run from the root folder:  uv run python live_transcribe_kroko.py
Stop with Ctrl+C.
"""

import shutil
from pathlib import Path

import numpy as np
import sherpa_onnx
from huggingface_hub import snapshot_download

from audio_capture import get_loopback, get_microphone, read_chunk, SAMPLERATE

# Model settings (the same model as in learning/sherpa-onnx-test.ipynb)
KROKO_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
NUM_THREADS = 4
SOURCE = "loopback"   # "loopback" = the sound that goes to the speaker, "microphone" = the sound in the room
RULE3_SECONDS = 300   # endpoint after 300 s of speech without a pause (default 20 s)

SENTENCE_END = ".?!"
PUNCTUATION = ".,!?;:"


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
        rule3_min_utterance_length=RULE3_SECONDS,
    )


# Make one segment (the format of the transcript_segments table) from some tokens.
# offset is start_time of the stream segment: the token times start at 0 after every reset.
# Punctuation at the start is skipped: after a reset, the first token is sometimes the "." of the previous sentence.
# Returns None if there are no words. end_ms is the time of the last token: the last word can continue a little after it.
def make_segment(tokens, times, offset):
    first = 0
    while first < len(tokens) and (tokens[first].strip() == "" or tokens[first].strip() in PUNCTUATION):
        first += 1
    if first == len(tokens):
        return None
    return {
        "start_ms": round((offset + times[first]) * 1000),
        "end_ms": round((offset + times[-1]) * 1000),
        "text": "".join(tokens[first:]).strip(),  # every token starts with a space if it starts a word
    }


# Find the finished sentences in the tokens, from token number done.
# Returns the sentences and the new done (the first token that is not in a finished sentence).
def take_sentences(tokens, times, offset, done):
    sentences = []
    for i in range(done, len(tokens)):
        token = tokens[i].strip()
        if token and token[-1] in SENTENCE_END:
            segment = make_segment(tokens[done:i + 1], times[done:i + 1], offset)
            if segment:
                sentences.append(segment)
            done = i + 1
    return sentences, done


# Print the unfinished sentence on the last line. \r goes back to the start of the line, so the line is written again.
# If the text is longer than the terminal, only the end is shown (a line that wraps cannot be written again).
def show_partial(text):
    width = shutil.get_terminal_size().columns - 1
    if len(text) > width:
        text = "..." + text[-(width - 3):]
    print("\r" + text.ljust(width), end="", flush=True)


def print_sentence(segment):
    width = shutil.get_terminal_size().columns - 1
    print("\r" + " " * width + "\r", end="")  # clear the partial line
    print(f"[{segment['start_ms'] / 1000:6.1f} - {segment['end_ms'] / 1000:6.1f} s] {segment['text']}")


# Run the encoder if the stream has enough audio, then take the finished sentences.
# At an endpoint (or at the end of the input), the rest of the text is also a sentence, and the stream is reset.
# done is the number of tokens of this stream segment that are already in printed sentences.
def decode(recognizer, stream, done, last=False):
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    result = recognizer.get_result_all(stream)
    tokens = list(result.tokens)
    times = list(result.timestamps)
    sentences, done = take_sentences(tokens, times, result.start_time, done)

    partial = ""
    if last or recognizer.is_endpoint(stream):
        rest = make_segment(tokens[done:], times[done:], result.start_time)
        if rest:
            sentences.append(rest)
        recognizer.reset(stream)
        done = 0
    else:
        partial = "".join(tokens[done:]).strip().lstrip(PUNCTUATION + " ")

    for segment in sentences:
        print_sentence(segment)
    show_partial(partial)
    return done, sentences


# Give one chunk to the model.
def process_chunk(recognizer, stream, chunk, done):
    stream.accept_waveform(SAMPLERATE, chunk)
    return decode(recognizer, stream, done)


# When we stop, the last sentence has no endpoint yet. Add some silence (tail padding)
# so the model can finish the last words, then take the rest as the last sentence.
def finish(recognizer, stream, done):
    stream.accept_waveform(SAMPLERATE, np.zeros(int(0.66 * SAMPLERATE), dtype=np.float32))
    stream.input_finished()
    return decode(recognizer, stream, done, last=True)


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

    done = 0
    # blocksize = 1 s of audio: the Windows buffer keeps 1 s while the model works.
    # The default is only about 10 ms, so audio was lost ("data discontinuity in recording").
    with source.recorder(samplerate=SAMPLERATE, blocksize=SAMPLERATE) as recorder:
        try:
            while True:
                chunk = read_chunk(recorder)
                done, _ = process_chunk(recognizer, stream, chunk, done)
        except KeyboardInterrupt:
            finish(recognizer, stream, done)
            print("\nStopped.")


if __name__ == "__main__":
    main()
