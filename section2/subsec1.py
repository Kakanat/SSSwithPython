from types import SimpleNamespace
import wave as wave
import pyroomacoustics as pa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

pa.datasets.CMUArcticCorpus(basedir="../CMU_ARCTIC", download=True, speaker=["aew", "axb"])
sample_wave_file="../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"

wav = wave.open(sample_wave_file)
print("sampling rate[Hz]: ", wav.getframerate())
print("sample size[Byte]: ", wav.getsampwidth())
print("num of samples: ", wav.getnframes())
print("num of channels: ", wav.getnchannels())
data = wav.readframes(wav.getnframes())
data = np.frombuffer(data, dtype=np.int16)
wav.close()

# make the figure of a sound file
data = data / np.iinfo(np.int16).max
x = np.array(range(wav.getnframes())) / wav.getframerate()
plt.figure(figsize=(10,4))
plt.xlabel("Time [sec]")
plt.ylabel("Value [-1,1]")
plt.plot(x, data)
plt.savefig("./results/wave_form.png")
plt.show()

# make a figure of white noise
n_sample = 40000
sample_rate = 16000
np.random.seed(0)
data = np.random.normal(size=n_sample)
x = np.array(range(n_sample)) / sample_rate
plt.figure(figsize=(10,4))
plt.xlabel("Time [sec]")
plt.ylabel("Value")
plt.plot(x, data)
plt.savefig("./results/white_noise.png")
plt.show()

# wtite sound data into a file
np.random.seed(0)
data = np.random.normal(scale=0.1, size=n_sample)
data_scale_adjust = data * np.iinfo(np.int16).max
data_scale_adjust = data_scale_adjust.astype(np.int16)
wave_out = wave.open("./results/wgn_wave.wav", 'w')
wave_out.setnchannels(1)
wave_out.setsampwidth(2)
wave_out.setframerate(sample_rate)
wave_out.writeframes(data_scale_adjust)
wave_out.close()

# play the sound
sd.play(data, wav.getframerate())
print("start playing")
status = sd.wait()