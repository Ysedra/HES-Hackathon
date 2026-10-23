"""
Transcription service: audio in, timestamped transcript segments out.

One algorithm for every input:
- a complete file or browser recording (mp3, WebM/Opus, MP4, Ogg...): transcribe_file()
- the live sound of the computer or the microphone (Windows only): transcribe_live()

Both feed audio to a StreamTranscriber, which transcribes it in overlapping windows.
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

log = logging.getLogger("transcription")


@dataclass
class Segment:
    start_ms: int   # from the start of the audio
    end_ms: int
    text: str


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


def _worker_main(path: str) -> None:
    """JSON-lines worker used only by transcribe_file_isolated()."""
    model = load_model()
    for segment in transcribe_file(model, path):
        print(json.dumps({"segment": segment.__dict__}), flush=True)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    _worker_main(sys.argv[2])
