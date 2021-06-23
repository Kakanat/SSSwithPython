import wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import matplotlib.pyplot as plt

np.random.seed(0)
clean_wave_file_test = "../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"

# wav = wave.open(clean_wave_file_test)
# data = wav.readframes(wav.getnframes())
# data = np.frombuffer(data, dtype=np.int16)
# data = data / np.iinfo(np.int16).max
# wav.close()
# sample_rate = 16000
# n_impulse_length = 512
# impulse_response = np.random.normal(size=n_impulse_length)
# conv_data = signal.convolve(data, impulse_response, mode='full')
# data /= data.max()
# data *= np.iinfo(np.int16).max
# data = data.astype(np.int16)
# sd.play(data, wav.getframerate())
# print("start playing")
# status = sd.wait()
# conv_data /= conv_data.max()
# conv_data *= np.iinfo(np.int16).max
# conv_data = conv_data.astype(np.int16)
# sd.play(conv_data, wav.getframerate())
# print("start playing")
# status = sd.wait()

clean_wave_files = ["../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav", "../CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0002.wav"]
n_sources = len(clean_wave_files)
n_samples = 0
for clean_wave_file in clean_wave_files:
    wav = wave.open(clean_wave_file)
    if n_samples < wav.getnframes():
        n_samples = wav.getnframes()
    wav.close()
clean_data = np.zeros([n_sources, n_samples])
s = 0
for clean_wave_file in clean_wave_files:
    wav = wave.open(clean_wave_file)
    data = wav.readframes(wav.getnframes())
    data = np.frombuffer(data, dtype=np.int16)
    data = data / np.iinfo(np.int16).max
    clean_data[s, :wav.getnframes()] = data
    wav.close()
    s += 1
sample_rate = 16000
SNR = 90
# SNR = 10
room_dim = np.r_[10.0, 10.0, 10.0]
mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1
mic_alignments = np.array([[-0.01, 0.0, 0.0],
                           [ 0.01, 0.0, 0.0]])
n_channels = np.shape(mic_alignments)[0]
R = mic_alignments.T + mic_array_loc[:, None]
print(R.T)
# room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=30, absorption=0.2)
room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
doas = np.array([[np.pi / 2., 0.],
                 [np.pi / 2., np.pi / 2.]])
distance = 1.
source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance
source_locations += mic_array_loc[:, None]
for s in range(n_sources):
    clean_data[s] /=  np.std(clean_data[s])
    room.add_source(source_locations[:, s], signal=clean_data[s])
room.simulate(snr=SNR)

impulse_response = room.rir
rt60 = pa.experimental.measure_rt60(impulse_response[0][0], fs=sample_rate)
print(f"Reverberation Time: {rt60} [sec]")

rir_power=np.square(impulse_response[0][0])
impulse_length=np.shape(impulse_response[0][0])[0]
reverb_power=np.zeros_like(rir_power)
for t in range(impulse_length):
    reverb_power[t]=10.*np.log10(np.sum(rir_power[t:])/np.sum(rir_power))
#x軸の値
x=np.array(range(impulse_length))/sample_rate
#音声データをプロットする
plt.figure(figsize=(10,4))
#x軸のラベル
plt.xlabel("Time [sec]")
#y軸のラベル
plt.ylabel("Value")
#x軸の範囲を設定する
plt.xlim([0,0.5])
#データをプロット
plt.plot(x,impulse_response[0][0])
# plt.plot(x,reverb_power)
#画像を画面に表示
plt.show()


# data1 = room.mic_array.signals[0]
# data1 = data1 * np.iinfo(np.int16).max / 20.
# data1 = data1.astype(np.int16)
# data2 = room.mic_array.signals[1]
# data2 = data2 * np.iinfo(np.int16).max / 20.
# data2 = data2.astype(np.int16)
# # print(room.mic_array.signals.shape) # (2, 62210)
# fig = plt.figure(figsize=(10,4))
# spectrum, freqs, t, im = plt.specgram(data2, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="gray")
# fig.colorbar(im).set_label("Intensity [dB]")
# plt.xlabel("Time [sec]")
# plt.ylabel("Frequency [Hz]")
# plt.show()
# sd.play(data2, wav.getframerate())
# print("start playing")
# status = sd.wait()