"""
Capture the audio that is sent to the default speaker (WASAPI loopback).
It is working on windows.

https://soundcard.readthedocs.io/en/latest/#soundcard._Microphone
https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording
https://learn.microsoft.com/en-us/windows/win32/coreaudio/wasapi


"""

import numpy as np
import sounddevice as sd
import soundcard as sc


RECORD_SECONDS = 3

speaker = sc.default_speaker()

loopback = sc.get_microphone(id=speaker.name,
                             include_loopback=True)

device = sd.query_devices(speaker.name + ", Windows WASAPI", kind="output")
samplerate = int(device["default_samplerate"])

print("Speaker:    ", speaker.name)
print("Loopback:   ", loopback)
print("Sample rate:", samplerate)

data = loopback.record(samplerate=samplerate,
                       numframes=samplerate * RECORD_SECONDS)

print("Shape:", data.shape)
print("Type: ", data.dtype)
print("Max:  ", np.abs(data).max())
