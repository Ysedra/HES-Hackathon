"""
- Live transcription with the Kroko streaming model (sherpa-onnx) of the audio that the computer plays (WASAPI loopback) or of the microphone (SOURCE setting). Works on Windows only.
The audio comes from audio_capture.py in small chunks, and every chunk goes to the model at once.

Differences from live_transcribe.py (Whisper):
- No buffer, no overlap, no queue and no second thread. Kroko is a streaming model: it keeps its own state (the stream) between chunks.
- The encoder runs when the stream has 1.28 s of new audio. One run takes much less time than the next 0.5 s chunk, so one thread is enough.
- With greedy search a token is never changed after the model gives it.

Words: the output is word by word, with start_ms, end_ms and text (one word = one row of transcript_segments).
We do not make sentences here: the backend does it (SentenceAssembler in webapp/backend/services/extraction.py).
- Kroko gives tokens (parts of words), not words. A token that starts with a space starts a new word.
  So a word is finished when the first token of the next word comes ("infla" can still get "tion").
- Punctuation stays in the word ("fly.", "true,"), so the SentenceAssembler can find the end of the sentence.
- Kroko gives one time per token, not an end time. end_ms = the time of the last token of the word + END_PAD_MS,
  but not later than the start of the next word. So end_ms is an estimate.
- The times are from the start of the stream (when this script started).

Pause: when the speaker is silent for some time, the model gives an endpoint. Then the last word is finished too,
the stream is reset, and process_chunk returns pause=True. The backend should call SentenceAssembler.flush() then:
pysbd keeps the last sentence until the next sentence starts, and in a pause this can take many seconds.
Rule 1: 2.4 s of silence (also when nothing was said), rule 2: 1.2 s of silence after speech,
rule 3: the segment is longer than RULE3_SECONDS. With the default 20 s, rule 3 cut words in half ("epis" / "ode").

Example for the backend:
    done, words, pause = process_chunk(recognizer, stream, chunk, done)
    for w in words:   # insert w into transcript_segments, then:
        sentences += assembler.add(segment_id, w["start_ms"], w["end_ms"], w["text"])
    if pause:
        sentences += assembler.flush()

Run from the root folder:  uv run python live_transcribe_kroko.py
Stop with Ctrl+C.
"""

import sys
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
END_PAD_MS = 200      # end_ms = time of the last token of the word + this (estimate)

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


def is_punctuation(token):
    return token.strip() == "" or token.strip() in PUNCTUATION


# Find the finished words in the tokens, from token number done.
# offset is start_time of the stream segment: the token times start at 0 after every reset.
# last=True: the stream segment ends, so the last word is finished too.
# Returns the words and the new done (the first token of the first unfinished word).
def take_words(tokens, times, offset, done, last=False):
    # After a reset, the first token is sometimes the "." of the previous sentence.
    # That word is already sent, so we skip it.
    if done == 0:
        while done < len(tokens) and is_punctuation(tokens[done]):
            done += 1

    # starts: the token numbers where a word starts
    starts = [i for i in range(done, len(tokens)) if i == done or tokens[i].startswith(" ")]
    if not last and starts:
        starts_finished = starts[:-1]  # the last word can still get more tokens
    else:
        starts_finished = starts

    words = []
    for n, a in enumerate(starts_finished):
        b = starts[n + 1] if n + 1 < len(starts) else len(tokens)  # the tokens of this word: a .. b-1
        end = times[b - 1] + END_PAD_MS / 1000
        if b < len(tokens):
            end = min(end, times[b])
        words.append({
            "start_ms": round((offset + times[a]) * 1000),
            "end_ms": round((offset + end) * 1000),
            "text": "".join(tokens[a:b]).strip(),
        })
    if starts_finished:
        done = starts[len(starts_finished)] if len(starts_finished) < len(starts) else len(tokens)
    return words, done


# Run the encoder if the stream has enough audio, then take the finished words.
# At an endpoint (or with last=True at the end of the input), the stream is reset and pause is True.
# done is the number of tokens of this stream segment that are already in finished words.
def decode(recognizer, stream, done, last=False):
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    result = recognizer.get_result_all(stream)
    endpoint = last or recognizer.is_endpoint(stream)
    words, done = take_words(list(result.tokens), list(result.timestamps), result.start_time, done, last=endpoint)

    pause = False
    if endpoint:
        pause = len(result.tokens) > 0  # only a pause after speech, not in a long silence
        recognizer.reset(stream)
        done = 0
    return done, words, pause


# Give one chunk to the model.
def process_chunk(recognizer, stream, chunk, done):
    stream.accept_waveform(SAMPLERATE, chunk)
    return decode(recognizer, stream, done)


# When we stop, the last word has no endpoint yet. Add some silence (tail padding)
# so the model can finish the last words, then take the rest.
def finish(recognizer, stream, done):
    stream.accept_waveform(SAMPLERATE, np.zeros(int(0.66 * SAMPLERATE), dtype=np.float32))
    stream.input_finished()
    return decode(recognizer, stream, done, last=True)


# Print the words on one line, and a new line at a pause. A line starts with the time of its first word.
def print_words(words, pause, line_open):
    for w in words:
        if not line_open:
            print(f"[{w['start_ms'] / 1000:6.1f} s]", end="")
            line_open = True
        print(" " + w["text"], end="", flush=True)
    if pause and line_open:
        print()
        line_open = False
    return line_open


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
    line_open = False
    # blocksize = 1 s of audio: the Windows buffer keeps 1 s while the model works.
    # The default is only about 10 ms, so audio was lost ("data discontinuity in recording").
    # On macOS blocksize means something else and must be small (5-320 on one Mac),
    # so 1 s gives a TypeError there: we use the default of the device.
    blocksize = SAMPLERATE if sys.platform == "win32" else None
    with source.recorder(samplerate=SAMPLERATE, blocksize=blocksize) as recorder:
        try:
            while True:
                chunk = read_chunk(recorder)
                done, words, pause = process_chunk(recognizer, stream, chunk, done)
                line_open = print_words(words, pause, line_open)
        except KeyboardInterrupt:
            done, words, pause = finish(recognizer, stream, done)
            print_words(words, True, line_open)
            print("Stopped.")


if __name__ == "__main__":
    main()
