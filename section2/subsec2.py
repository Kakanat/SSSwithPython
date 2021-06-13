from types import SimpleNamespace
import wave as wave
import pyroomacoustics as pa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sp

wave_length = 5
sample_rate = 16000
print("Start recording.")
data = sd.rec(int(wave_length * sample_rate), sample_rate, channels=1)
sd.wait()

sample_wave_file="../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"
wav = wave.open(sample_wave_file)
print("sampling rate[Hz]: ", wav.getframerate())
print("sample size[Byte]: ", wav.getsampwidth())
print("num of samples: ", wav.getnframes())
print("num of channels: ", wav.getnchannels())
data = wav.readframes(wav.getnframes())
data = np.frombuffer(data, dtype=np.int16)
wav.close()

f,t,stft_data = sp.stft(data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
print("shape after STFT:", np.shape(stft_data))
print("axis of F [Hz]:", f)
print("axis of time [sec]:", t)

