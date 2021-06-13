from types import SimpleNamespace
import wave as wave
import pyroomacoustics as pa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sp

sample_wave_file="../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"
wav = wave.open(sample_wave_file)
print("sampling rate[Hz]: ", wav.getframerate())
print("sample size[Byte]: ", wav.getsampwidth())
print("num of samples: ", wav.getnframes())
print("num of channels: ", wav.getnchannels())
data = wav.readframes(wav.getnframes())
data = np.frombuffer(data, dtype=np.int16)
wav.close()

fig = plt.figure(figsize=(10,4))
spectrum, freqs, t, im = plt.specgram(data, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="jet")
fig.colorbar(im).set_label("Intensity [dB]")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.savefig("./results/spectrogram.png")
plt.show()