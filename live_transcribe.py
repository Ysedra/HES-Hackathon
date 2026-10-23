"""
- Live transcription of the audio that the computer plays (WASAPI loopback) or of the microphone (SOURCE setting). Works on Windows only.
The audio comes from audio_capture.py in small chunks. The chunks are collected in a buffer, and when the buffer is full, it is given to faster-whisper.

Two threads work at the same time:
- The main thread reads the loopback without a break and fills the buffer. When the buffer is full, it puts the buffer into a queue.
- The Whisper thread takes the buffers from the queue and transcribes them.
If both jobs were in one thread, the audio would be lost while Whisper works (the "data discontinuity in recording" warning).

Overlap: every buffer starts with the last OVERLAP_SECONDS of the previous buffer. So a word that is cut at the end of one buffer is complete in the next buffer.
Two rules with the word timestamps decide which words are printed:
- Rule 1: a word that ends in the last OVERLAP_SECONDS / 2 of the buffer can be cut, so it is not printed. The next buffer has it.
- Rule 2: a word that was already printed (in the overlap) is not printed again.

Real-time condition: the transcription of one buffer must take less time than BUFFER_SECONDS. If not, the queue grows and the text comes later and later.
The number in the "queue" output shows this. It should stay at 0.

Settings that we tested (CPU, Ryzen 7 PRO 5850U):
- BUFFER_SECONDS = 5 had no safety margin: about 4.5 s for every buffer, one slow buffer made the queue grow to 7.
- BUFFER_SECONDS = 10 with beam_size=5 worked, but the delay was 10-17 s.
- temperature=0.0 turns off the temperature fallback. Without it, one bad buffer took 38.7 s (it was decoded again and again).

Run from the root folder:  uv run python live_transcribe.py
Stop with Ctrl+C.
"""

import queue
import threading
import time

import numpy as np
from faster_whisper import WhisperModel

from audio_capture import get_loopback, get_microphone, read_chunk, SAMPLERATE, CHUNK_SECONDS

# Model settings (the same model as in the podcast test)
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
CPU_THREADS = 8
LANGUAGE = "en"   # must be the language of the video, a wrong language is very slow
BEAM_SIZE = 2    # 1 = greedy, faster than 5
SOURCE = "loopback"   # "loopback" = the sound that goes to the speaker, "microphone" = the sound in the room

# Buffer settings
BUFFER_SECONDS = 8    # new audio in each buffer
OVERLAP_SECONDS = 2   # audio taken again from the end of the previous buffer


def load_model():
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)


# Choose the words of one buffer that can be printed (rule 1 and rule 2).
# The word times from Whisper are from the start of the buffer, so we add buffer_start
# to get the time from the start of the stream.
def select_new_words(words, buffer_start, cutoff, last_end):
    new_words = []
    for w in words:
        w_start = buffer_start + w.start
        w_end = buffer_start + w.end
        if (w_start + w_end) / 2 < last_end:  # rule 2: already printed
            continue
        if w_end > cutoff:                    # rule 1: maybe cut, the next buffer has it
            break
        new_words.append(w.word)
        last_end = w_end
    # faster-whisper puts a space before every word, so we join with "".
    return "".join(new_words).strip(), last_end


# The Whisper thread: take a buffer from the queue, transcribe it, print the new words.
def transcribe_worker(model, audio_queue):
    last_end = 0.0  # end time of the last printed word (seconds from the start of the stream)
    while True:
        item = audio_queue.get()  # waits until a buffer is in the queue
        if item is None:          # None means: stop
            break
        buffer, buffer_start = item
        buffer_end = buffer_start + len(buffer) / SAMPLERATE
        cutoff = buffer_end - OVERLAP_SECONDS / 2

        start = time.perf_counter()
        segments, _ = model.transcribe(buffer, language=LANGUAGE, beam_size=BEAM_SIZE, temperature=0.0,
                                       vad_filter=True, word_timestamps=True)
        # Put the words of all segments in one list. The real work happens in this loop.
        words = []
        for segment in segments:
            for word in segment.words:
                words.append(word)
        elapsed = time.perf_counter() - start

        text, last_end = select_new_words(words, buffer_start, cutoff, last_end)
        print(f"({elapsed:.1f} s, queue: {audio_queue.qsize()}) {text}")


# The main thread: read the source (loopback or microphone), fill the buffer, put it into the queue.
def main():
    print("Loading the model...")
    model = load_model()

    audio_queue = queue.Queue()
    worker = threading.Thread(target=transcribe_worker, args=(model, audio_queue), daemon=True)
    worker.start()

    n_chunks = round(BUFFER_SECONDS / CHUNK_SECONDS)
    n_overlap = int(OVERLAP_SECONDS * SAMPLERATE)
    stream_time = 0.0                              # seconds read since the start
    previous_tail = np.zeros(0, dtype=np.float32)  # last OVERLAP_SECONDS of the previous buffer

    if SOURCE == "microphone":
        source = get_microphone()
    else:
        source = get_loopback()

    print("Recording from:", source)
    print(f"Buffer: {BUFFER_SECONDS} s + {OVERLAP_SECONDS} s overlap, beam_size: {BEAM_SIZE}, language: {LANGUAGE}")
    print("Press Ctrl+C to stop.")

    with source.recorder(samplerate=SAMPLERATE) as recorder:
        try:
            while True:
                # Read n_chunks chunks and keep them in a list
                chunks = []
                for i in range(n_chunks):
                    chunk = read_chunk(recorder)
                    chunks.append(chunk)
                new_audio = np.concatenate(chunks)

                buffer = np.concatenate([previous_tail, new_audio])           # overlap + new audio
                buffer_start = stream_time - len(previous_tail) / SAMPLERATE  # where this buffer starts in the stream
                audio_queue.put((buffer, buffer_start))  # give it to Whisper, do not wait

                previous_tail = new_audio[-n_overlap:]
                stream_time += len(new_audio) / SAMPLERATE
        except KeyboardInterrupt:
            print("Stopping...")

    audio_queue.put(None)  # tell the Whisper thread to stop
    worker.join()          # wait until it finishes the buffers in the queue
    print("Stopped.")


if __name__ == "__main__":
    main()
