from types import SimpleNamespace
import wave as wave
import pyroomacoustics as pa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sp

sample_wave_file="../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"
wav = wave.open(sample_wave_file)
data = wav.readframes(wav.getnframes())
data = np.frombuffer(data, dtype=np.int16)
wav.close()

f,t,stft_data = sp.stft(data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)

t, data_post = sp.istft(stft_data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
data_post = data_post.astype(np.int16)
wave_out = wave.open("./results/istft_post_wave.wav", 'w')
wave_out.setnchannels(1)
wave_out.setsampwidth(2)
wave_out.setframerate(wav.getframerate())
wave_out.writeframes(data_post)
wave_out.close()

sd.play(data_post, wav.getframerate())
print("start playing")
status = sd.wait()