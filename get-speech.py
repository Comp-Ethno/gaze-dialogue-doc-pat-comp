import wave
import contextlib
import numpy as np
import pandas as pd
import math
import webrtcvad

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        print (timestamp)
        offset += n

speech_ms =[]
audio, sample_rate = read_wave('audio/sample.wav')
vad = webrtcvad.Vad(3)
frames = frame_generator(10, audio, sample_rate)
frames = list(frames);

for frame in frames:
    speech_ms.append (vad.is_speech(frame.bytes, sample_rate))

speech_ms = np.array(speech_ms).astype(int)
n=100
speech = [math.ceil(sum(speech_ms[i:i+n])/n) for i in range(0,len(speech_ms),n)]

clock = np.arange(0, len(speech)*0.5, 0.5)
result_speech = pd.DataFrame ({'clock':clock, 'speech':speech})
pd.DataFrame(result_speech).to_csv('classifications/speech-sample.csv', index=False)