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

# f,t,stft_data = sp.stft(data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
# stft_data[100:,:] = 0
# t, data_post = sp.istft(stft_data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
# data_post = data_post.astype(np.int16)
# sd.play(data_post, wav.getframerate())
# print("start playing")
# status = sd.wait()
# fig = plt.figure(figsize=(10,4))
# spectrum, freqs, t, im = plt.specgram(data_post, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="jet")
# fig.colorbar(im).set_label("Intensity [dB]")
# plt.xlabel("Time [sec]")
# plt.ylabel("Frequency [Hz]")
# plt.show()

# f,t,stft_data = sp.stft(data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
# stft_data[:50,:] = 0
# t, data_post = sp.istft(stft_data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
# data_post = data_post.astype(np.int16)
# sd.play(data_post, wav.getframerate())
# print("start playing")
# status = sd.wait()
# fig = plt.figure(figsize=(10,4))
# spectrum, freqs, t, im = plt.specgram(data_post, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="jet")
# fig.colorbar(im).set_label("Intensity [dB]")
# plt.xlabel("Time [sec]")
# plt.ylabel("Frequency [Hz]")
# plt.show()

n_speech = wav.getnframes()
n_noise_only = 40000
n_sample = n_noise_only + n_speech
wgn_signal = np.random.normal(scale=0.04, size=n_sample)
wgn_signal = wgn_signal * np.iinfo(np.int16).max
wgn_signal = wgn_signal.astype(np.int16)
mix_signal = wgn_signal
mix_signal[n_noise_only:] += data
sd.play(mix_signal, wav.getframerate())
print("start playing")
status = sd.wait()
fig = plt.figure(figsize=(10,4))
spectrum, freqs, t, im = plt.specgram(mix_signal, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="gray")
fig.colorbar(im).set_label("Intensity [dB]")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.show()
f,t,stft_data = sp.stft(mix_signal, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)

amp = np.abs(stft_data)
phase = stft_data / np.maximum(amp, 1.e-20)
n_noise_only_frame = np.sum(t < (n_noise_only / wav.getframerate()))
p = 1.0
alpha = 20.0
noise_amp = np.power(np.mean(np.power(amp, p)[:, :n_noise_only_frame], axis=1, keepdims=True), 1./2)
eps = 0.01 * np.power(amp, p)
processed_amp = np.power(np.maximum(np.power(amp, p) - alpha * np.power(noise_amp, p), eps), 1./p)
processed_stft_data = processed_amp * phase
t, processed_data_post = sp.istft(processed_stft_data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
processed_data_post = processed_data_post.astype(np.int16)
sd.play(processed_data_post, wav.getframerate())
print("start playing")
status = sd.wait()
fig = plt.figure(figsize=(10,4))
spectrum, freqs, t, im = plt.specgram(processed_data_post, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="gray")
fig.colorbar(im).set_label("Intensity [dB]")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.show()

amp = np.abs(stft_data)
input_power = np.power(amp, 2.0)
n_noise_only_frame = np.sum(t < (n_noise_only / wav.getframerate()))
alpha = 1.0
mu = 20
noise_power = np.mean(np.power(amp, 2.0)[:, :n_noise_only_frame], axis=1, keepdims=True)
eps = 0.01 * input_power
processed_power = np.maximum(input_power - alpha * noise_power, eps)
wf_ratio = processed_power / (processed_power + mu * noise_power)
processed_stft_data = wf_ratio * stft_data
t, processed_data_post = sp.istft(processed_stft_data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
processed_data_post = processed_data_post.astype(np.int16)
sd.play(processed_data_post, wav.getframerate())
print("start playing")
status = sd.wait()
fig = plt.figure(figsize=(10,4))
spectrum, freqs, t, im = plt.specgram(processed_data_post, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="gray")
fig.colorbar(im).set_label("Intensity [dB]")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.show()