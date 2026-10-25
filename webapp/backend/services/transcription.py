"""
Transcription service: audio in, timestamped transcript segments out.

Two models, one per kind of input:
- a complete file or browser recording (mp3, WebM/Opus, MP4, Ogg...): Whisper, transcribe_file()
- the live sound of the computer (loopback) or the microphone: the Kroko streaming model,
  transcribe_live_kroko() (from live_transcribe_kroko.py). transcribe_live() is the older Whisper
  version of live capture; it is kept for comparison but no route uses it.

Whisper is fed by a StreamTranscriber, which transcribes the audio in overlapping windows.
Each window owns the words that start in its first window_s - overlap_s seconds; the overlap
belongs to the next window, which hears those words with more context. Whisper's word
timestamps let a window skip words the previous window already returned, so no words are
repeated or lost at a boundary (same rules as pipeline_v0.ipynb and live_transcribe.py).

The service knows nothing about HTTP or the database: the pipeline decides what to do with
the segments.
"""

import logging
import json
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Iterator

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

SAMPLE_RATE = 16000   # Whisper expects 16 kHz mono audio
LANGUAGE = "en"       # must be the language of the audio, a wrong language is very slow

# Model settings (the same model as in the podcast test)
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
CPU_THREADS = 8

# Files are not real time, so they use longer windows and a wider beam (pipeline_v0.ipynb).
FILE_SETTINGS = dict(window_s=30, overlap_s=5, beam_size=5)
# Live audio must be transcribed faster than it arrives (tested in live_transcribe.py):
# 8 s of new audio + 2 s overlap, and temperature=0.0 so one bad window isn't decoded again and again.
LIVE_SETTINGS = dict(window_s=10, overlap_s=2, beam_size=2, temperature=0.0)
CAPTURE_CHUNK_S = 0.5   # live audio is read in chunks of this length

# Live mode: Kroko, a streaming Zipformer transducer (sherpa-onnx), tested in asr/sherpa-onnx-test.ipynb.
# It writes text while the audio arrives (no window to wait for), about 7x faster than Whisper small.
KROKO_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
KROKO_THREADS = 4
KROKO_RULE3_S = 300      # endpoint after 300 s of speech without a pause (the default 20 s cut words in half)
# Kroko often leaves out the full stop when the speaker doesn't pause. A long live sentence is then
# cut at a natural place: at a comma after LIVE_COMMA_WORDS words, else before "and", "but" or "so"
# after LIVE_CONJUNCTION_WORDS words (not earlier: "Trinidad and Tobago" stays whole), and in any
# case after LIVE_MAX_WORDS words. "which", "who", "that", "because"... are never cut before.
LIVE_COMMA_WORDS = 20
LIVE_CONJUNCTION_WORDS = 25
LIVE_CUT_BEFORE = {"and", "but", "so"}
LIVE_MAX_WORDS = 40
LIVE_END_PAD_MS = 200    # Kroko gives one time per token, not an end time: end_ms is an estimate
LIVE_HEARTBEAT_S = 1.0   # with no audio for this long, a Pause is still yielded (see Pause)
SENTENCE_END = ".?!"
PUNCTUATION = ".,!?;:"

log = logging.getLogger("transcription")


@dataclass
class Segment:
    start_ms: int   # from the start of the audio
    end_ms: int
    text: str


@dataclass
class Partial:
    """Live only, when asked for: the unfinished sentence so far, shown grey until it ends.
    An empty text means the sentence was finished (or dropped) and the grey text can go."""
    start_ms: int
    text: str


@dataclass
class Pause:
    """Live only: a sentence ended, the speaker paused, or no audio came for LIVE_HEARTBEAT_S.
    The pipeline completes its sentences at a Pause, and the regular Pauses let a live run
    notice a closed browser or a stop request even when nobody speaks."""


def load_model() -> "WhisperModel":
    # Lazy import is essential on Intel macOS: merely importing CTranslate2 loads its copy
    # of libiomp5, which later collides with PyTorch even if Whisper runs in a child process.
    from faster_whisper import WhisperModel

    return WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)


def needs_isolated_worker() -> bool:
    """Intel macOS wheels for PyTorch and CTranslate2 bundle separate OpenMP runtimes.

    Loading both models in one process aborts Python with OpenMP error #15. Keep Whisper in
    a child process on that platform; Linux, Windows and Apple Silicon retain the normal path.
    """
    return sys.platform == "darwin" and platform.machine() == "x86_64"


def transcribe_file_isolated(source: BinaryIO, name: str) -> Iterator[Segment]:
    """Transcribe in a child process so CTranslate2 cannot collide with PyTorch's OpenMP."""
    suffix = Path(name).suffix if Path(name).suffix else ".media"
    backend_dir = Path(__file__).resolve().parents[1]

    with tempfile.NamedTemporaryFile(suffix=suffix) as media:
        shutil.copyfileobj(source, media)
        media.flush()
        command = [sys.executable, "-u", "-m", "services.transcription", "--worker", media.name]
        process = subprocess.Popen(
            command,
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                message = json.loads(line)
                yield Segment(**message["segment"])
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Whisper worker exited with status {return_code}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


class StreamTranscriber:
    """Transcribe audio that arrives in pieces, in overlapping windows.

    feed() takes new 16 kHz float32 audio and returns the segments it completes.
    Call flush() at the end of the audio to transcribe what is left."""

    def __init__(self, model, window_s, overlap_s, beam_size, temperature=None):
        self.model = model
        self.window = int(window_s * SAMPLE_RATE)
        self.step = int((window_s - overlap_s) * SAMPLE_RATE)
        self.options = dict(language=LANGUAGE, task="transcribe", beam_size=beam_size,
                            vad_filter=True, word_timestamps=True)
        if temperature is not None:
            self.options["temperature"] = temperature
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start = 0       # position of buffer[0] in the audio, in samples
        self.last_end_s = 0.0       # end of the last returned word, in audio time

    def feed(self, audio) -> list[Segment]:
        self.buffer = np.concatenate([self.buffer, np.asarray(audio, dtype=np.float32)])
        segments = []
        while len(self.buffer) >= self.window:
            segments += self._transcribe(self.buffer[:self.window], owned=self.step)
            self.buffer = self.buffer[self.step:]
            self.buffer_start += self.step
        return segments

    def flush(self) -> list[Segment]:
        segments = []
        if len(self.buffer):   # the last window owns all its words
            segments = self._transcribe(self.buffer, owned=len(self.buffer))
        self.buffer = np.zeros(0, dtype=np.float32)
        return segments

    def _transcribe(self, clip, owned) -> list[Segment]:
        start_s = self.buffer_start / SAMPLE_RATE
        owned_end_s = start_s + owned / SAMPLE_RATE
        started = time.perf_counter()
        whisper_segments, _ = self.model.transcribe(clip, **self.options)

        segments = []
        for s in whisper_segments:
            words = [
                w for w in (s.words or [])
                if start_s + w.start < owned_end_s                          # the next window owns the rest
                and start_s + (w.start + w.end) / 2 >= self.last_end_s      # not already returned
            ]
            # faster-whisper puts a space before every word, so we join with "".
            text = "".join(w.word for w in words).strip()
            if not text:
                continue
            segments.append(Segment(round((start_s + words[0].start) * 1000),
                                    round((start_s + words[-1].end) * 1000), text))
            self.last_end_s = max(self.last_end_s, start_s + words[-1].end)
        # The segments are a generator: Whisper's work happens in the loop above, so time it here
        log.info("window %7.1f-%7.1f s: %d segments in %.1f s", start_s,
                 start_s + len(clip) / SAMPLE_RATE, len(segments), time.perf_counter() - started)
        return segments


def transcribe_file(model, source: str | BinaryIO, **settings) -> Iterator[Segment]:
    """Transcribe a complete file, yielding segments window by window.

    source is a path or a file-like object; PyAV decodes the actual format (mp3, WebM/Opus,
    MP4, Ogg...) and resamples it to 16 kHz mono, so browser recordings work as they are."""
    from faster_whisper import decode_audio

    audio = decode_audio(source, sampling_rate=SAMPLE_RATE)
    log.info("decoded %.1f s of audio", len(audio) / SAMPLE_RATE)
    if not len(audio):
        log.warning("the file contains no audio that PyAV could decode")
    transcriber = StreamTranscriber(model, **{**FILE_SETTINGS, **settings})
    for i in range(0, len(audio), transcriber.step):
        yield from transcriber.feed(audio[i:i + transcriber.step])
    yield from transcriber.flush()


def transcribe_live(model, stop: threading.Event, source="loopback", **settings) -> Iterator[Segment]:
    """Transcribe the sound of the computer ("loopback") or the microphone until stop is set.

    Two threads, as in live_transcribe.py: a capture thread reads the audio without a break,
    so no audio is lost while Whisper works on a window in this thread.
    The capture works like audio_capture.py: WASAPI loopback, resampled by Windows to 16 kHz."""
    import soundcard as sc   # Windows only, so imported here and not at the top

    chunks = queue.Queue()

    def capture():
        if source == "microphone":
            device = sc.default_microphone()
        else:   # loopback: like a microphone, but it records the sound that goes to the speaker
            device = sc.get_microphone(id=sc.default_speaker().name, include_loopback=True)
        try:
            with device.recorder(samplerate=SAMPLE_RATE) as recorder:
                while not stop.is_set():
                    chunk = recorder.record(numframes=int(SAMPLE_RATE * CAPTURE_CHUNK_S))
                    chunks.put(chunk.mean(axis=1))   # mono
        finally:
            chunks.put(None)   # None means: capture stopped

    threading.Thread(target=capture, daemon=True).start()
    transcriber = StreamTranscriber(model, **{**LIVE_SETTINGS, **settings})
    while (chunk := chunks.get()) is not None:
        yield from transcriber.feed(chunk)
    yield from transcriber.flush()


def load_kroko():
    """The Kroko recognizer. The model (~57 MB) is downloaded from Hugging Face the first time."""
    import sherpa_onnx
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(KROKO_REPO))
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / "encoder.onnx"),
        decoder=str(model_dir / "decoder.onnx"),
        joiner=str(model_dir / "joiner.onnx"),
        num_threads=KROKO_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        model_type="zipformer2",
        decoding_method="greedy_search",   # a token is never changed after the model gives it
        enable_endpoint_detection=True,
        rule3_min_utterance_length=KROKO_RULE3_S,
    )


def normalise_numbers(text):
    """Kroko writes numbers as words ("twenty six cantons"), but the web evidence uses digits
    ("26 cantons"), so the words are changed to digits (inverse text normalisation)."""
    from text_to_num import alpha2digit

    text = alpha2digit(text, "en")
    # alpha2digit writes years said as two numbers ("nineteen ninety nine") as "19 99"
    text = re.sub(r"\b(1[1-9]|20) ([1-9][0-9])\b", r"\1\2", text)
    # and "two point five million" as "2.5 1000000": keep the scale word
    for value, word in (("1000000000", "billion"), ("1000000", "million")):
        text = re.sub(rf"(\d) {value}\b", rf"\1 {word}", text)
    return text


def _is_punctuation(token):
    return token.strip() == "" or token.strip() in PUNCTUATION


def _kroko_sentences(recognizer, stream, done, last=False):
    """Decode what the stream has, and cut the new tokens into finished sentences.

    A sentence ends at a token ending with . ? or !, at a natural place in a long sentence (see
    LIVE_COMMA_WORDS), or at an endpoint (a pause in the speech, or last=True). done is the number of tokens of this stream
    segment that are already in sentences. Returns (segments, done, endpoint, partial); partial
    is the unfinished sentence as a Segment, or None."""
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    result = recognizer.get_result_all(stream)
    tokens, times, offset = list(result.tokens), list(result.timestamps), result.start_time
    endpoint = last or recognizer.is_endpoint(stream)

    def make(a, b):   # tokens a .. b-1 as a segment; token times start at 0 after every reset
        text = "".join(tokens[a:b]).strip()   # a token that starts a word starts with a space
        if not any(c.isalnum() for c in text):
            return None
        start_ms = round((offset + times[a]) * 1000)
        end_ms = round((offset + times[b - 1]) * 1000) + LIVE_END_PAD_MS
        return Segment(start_ms, end_ms, normalise_numbers(text))

    if done == 0:   # after a reset the first token is sometimes the "." of the previous sentence
        while done < len(tokens) and _is_punctuation(tokens[done]):
            done += 1
    def next_word(i):   # the word that starts at token i, or None while it may still grow ("an" -> "android")
        j = i + 1
        while j < len(tokens) and not tokens[j].startswith(" "):
            j += 1
        return "".join(tokens[i:j]).strip().strip(PUNCTUATION).lower() if j < len(tokens) else None

    segments, start, words = [], done, 0
    for i in range(done, len(tokens)):
        words += tokens[i].startswith(" ") or i == start
        token = tokens[i].strip()
        # Only cut where the next word has begun, so a word or a number like "2,400" is never split
        next_starts_word = i + 1 < len(tokens) and tokens[i + 1].startswith(" ")
        if (token[-1:] in tuple(SENTENCE_END)
                or (next_starts_word and (
                    (words >= LIVE_COMMA_WORDS and token.endswith(","))
                    or (words >= LIVE_CONJUNCTION_WORDS and next_word(i + 1) in LIVE_CUT_BEFORE)
                    or words >= LIVE_MAX_WORDS))):
            segments.append(make(start, i + 1))
            start, words = i + 1, 0
    partial = None
    if endpoint:
        if start < len(tokens):
            segments.append(make(start, len(tokens)))
        recognizer.reset(stream)
        start = 0
    elif start < len(tokens):
        partial = make(start, len(tokens))
    return [s for s in segments if s], start, endpoint, partial


def _start_com():
    """soundcard starts COM only in the thread that imported it. In any other thread (a later live
    run, a request thread) WASAPI fails with 0x800401f0 unless that thread starts COM itself.
    Returns the library to stop COM with, or None outside Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    # soundcard's own COM start (at its first import) fails if this thread started COM first,
    # so it is imported before; in the importing thread the call below only adds a reference.
    import soundcard   # noqa: F401
    ole = ctypes.windll.ole32
    ole.CoInitializeEx(None, 0)   # 0 = COINIT_MULTITHREADED
    return ole


def _find_live_device(source):
    """The device to record, as (kind, device). kind is "loopback" for the computer's own sound
    (WASAPI loopback on Windows, the BlackHole virtual device on macOS), or "microphone" when the
    microphone is asked for or there is no loopback (a Mac without BlackHole). COM must be started."""
    import soundcard as sc   # imported here: only live runs need it

    if source == "loopback":
        if sys.platform == "darwin":   # macOS has no loopback: BlackHole is read like a microphone
            blackhole = [mic for mic in sc.all_microphones() if "blackhole" in mic.name.lower()]
            if blackhole:
                return "loopback", blackhole[0]
            log.warning("BlackHole is not installed: the live run uses the microphone")
        else:   # loopback: like a microphone, but it records the sound that goes to the speaker
            try:
                return "loopback", sc.get_microphone(id=sc.default_speaker().name, include_loopback=True)
            except Exception as error:
                log.warning("no loopback device (%s): the live run uses the microphone", error)
    return "microphone", sc.default_microphone()


def live_source(source="loopback"):
    """What a live run would record, for the live page to warn about the microphone:
    {"source": "loopback" | "microphone" | None, "device": name, "platform": sys.platform, "error": ...}."""
    com = _start_com()
    try:
        kind, device = _find_live_device(source)
        return {"source": kind, "device": device.name, "platform": sys.platform}
    except Exception as error:   # no microphone at all, no permission...
        return {"source": None, "device": None, "platform": sys.platform, "error": str(error)}
    finally:
        if com is not None:
            com.CoUninitialize()


def transcribe_live_kroko(recognizer, stop: threading.Event, source="loopback",
                          partials=False) -> Iterator[Segment | Partial | Pause]:
    """Transcribe the sound of the computer ("loopback") or the microphone with Kroko until stop is set.

    Yields one Segment per sentence as soon as it is finished, and a Pause after it, at every
    pause in the speech, and every LIVE_HEARTBEAT_S without audio. With partials=True it also
    yields a Partial whenever the unfinished sentence changes. A capture thread reads the
    audio without a break, so no audio is lost while the pipeline saves segments or checks claims
    (sequential versions). The times are from the start of the capture.
    Loopback: WASAPI on Windows; on macOS the BlackHole virtual device (not tested). Without a
    loopback device the microphone is used (see _find_live_device)."""
    chunks = queue.Queue()
    capture_errors = []

    def capture():
        com = None
        try:
            com = _start_com()
            kind, device = _find_live_device(source)
            log.info("live capture (%s) from %s", kind, device)
            # blocksize = 1 s: Windows keeps 1 s of audio while this thread is busy.
            # The default (~10 ms) lost audio ("data discontinuity in recording").
            with device.recorder(samplerate=SAMPLE_RATE, blocksize=SAMPLE_RATE) as recorder:
                while not stop.is_set():
                    chunk = recorder.record(numframes=int(SAMPLE_RATE * CAPTURE_CHUNK_S))
                    chunks.put(chunk.mean(axis=1))   # mono
        except Exception as error:   # no device, no permission...: reported by the generator
            capture_errors.append(error)
        finally:
            if com is not None:
                com.CoUninitialize()
            chunks.put(None)   # None means: capture stopped

    threading.Thread(target=capture, name="live-capture", daemon=True).start()
    stream = recognizer.create_stream()
    done = 0
    shown = ""   # the partial text the client has now
    try:
        while True:
            try:
                chunk = chunks.get(timeout=LIVE_HEARTBEAT_S)
            except queue.Empty:   # loopback gives no audio while nothing plays
                yield Pause()
                continue
            if chunk is None:
                break
            stream.accept_waveform(SAMPLE_RATE, chunk)
            segments, done, endpoint, partial = _kroko_sentences(recognizer, stream, done)
            yield from segments
            text = partial.text if partial else ""
            if partials and text != shown:
                yield Partial(partial.start_ms if partial else 0, text)
                shown = text
            if segments or endpoint:
                yield Pause()
        # Stopped: a little silence (tail padding) lets the model finish the last words
        stream.accept_waveform(SAMPLE_RATE, np.zeros(int(0.66 * SAMPLE_RATE), dtype=np.float32))
        stream.input_finished()
        segments, _, _, _ = _kroko_sentences(recognizer, stream, done, last=True)
        yield from segments
        if partials and shown:
            yield Partial(0, "")
        yield Pause()
    finally:
        stop.set()   # also when the pipeline closes this generator early
    if capture_errors:
        raise RuntimeError(f"Live audio capture failed: {capture_errors[0]!r}") from capture_errors[0]


def _worker_main(path: str) -> None:
    """JSON-lines worker used only by transcribe_file_isolated()."""
    model = load_model()
    for segment in transcribe_file(model, path):
        print(json.dumps({"segment": segment.__dict__}), flush=True)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    _worker_main(sys.argv[2])
