"""
- Capture the audio that is sent to the default speaker (WASAPI loopback). Works on Windows only.
The audio is taken with WASAPI loopback while it is still PCM (digital samples), before it reaches the speaker. These PCM values will then be given to the ASR model.

The mixer works in the mix format of the device (GetMixFormat): float32 samples, with the sample rate of the device (usually 48000 or 44100 Hz, it depends on the hardware and driver).
The recording sample rate is 16000 Hz: Windows resamples the mixer output to 16000 Hz. Each chunk is a mono float32 array in [-1, 1], which is the input format of the ASR models.

16000 Hz because the ASR models (Whisper, Kroko) are trained on 16 kHz audio. It keeps frequencies up to 8000 Hz, which is enough for speech. (If we decide to use a different model, this value can be changed.)

0.5 s is only a starting value: smaller chunks give lower latency but a higher risk of buffer underruns, larger chunks are safer but add delay.


Note: I used Claude Code for help while coding. For example, I could not find on my own how to read the audio in chunks. I could only record for a fixed time, and when this time ended, the recording (the loopback) also stopped.
With recorder() and read_chunk(), the recording stays open and the audio is read in small chunks.

Useful sources to understand the concepts:
https://soundcard.readthedocs.io/en/latest/#soundcard._Microphone
https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording
https://learn.microsoft.com/en-us/windows/win32/coreaudio/wasapi
https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nf-audioclient-iaudioclient-getmixformat

"""

import numpy as np
import soundcard as sc

SAMPLERATE = 16000
# TODO: tune this value when we measure the latency of the live ASR pipeline. (Maybe)
CHUNK_SECONDS = 0.5


# Find the speaker that plays the sound now and open its loopback.
# Loopback is like a microphone, but it records the sound that goes to the speaker.
def get_loopback():
    speaker = sc.default_speaker()
    return sc.get_microphone(id=speaker.name, include_loopback=True)

# Open the default microphone of the computer (the sound in the room).
def get_microphone():
    return sc.default_microphone()


# Take one chunk (0.5 s) from the recorder and make it mono.
# After this, the chunk is ready to give to the ASR model.
def read_chunk(recorder):
    numframes = int(SAMPLERATE * CHUNK_SECONDS)
    chunk = recorder.record(numframes=numframes)
    return chunk.mean(axis=1)


# Only for testing: record until Ctrl+C and print the level of every chunk.
if __name__ == "__main__":
    loopback = get_loopback()
    print("Recording from:", loopback)
    print("Press Ctrl+C to stop.")

    with loopback.recorder(samplerate=SAMPLERATE) as recorder:
        try:
            while True:
                chunk = read_chunk(recorder)
                level = np.abs(chunk).max()
                print(chunk.shape, chunk.dtype, round(float(level), 4), "#" * int(level * 50))
        except KeyboardInterrupt:
            print("Stopped.")
