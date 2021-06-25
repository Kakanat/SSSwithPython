import numpy as np
import pyroomacoustics as pa
import wave
import scipy.signal as sp

def calculate_steering_vector(mic_alignments,source_locations,freqs,sound_speed=340,is_use_far=False):
    #マイク数を取得
    n_channels=np.shape(mic_alignments)[1]

    #音源数を取得
    n_source=np.shape(source_locations)[1]

    if is_use_far==True:
        #音源位置を正規化
        norm_source_locations=source_locations/np.linalg.norm(source_locations,2,axis=0,keepdims=True)

        #位相を求める
        steering_phase=np.einsum('k,ism,ism->ksm',2.j*np.pi/sound_speed*freqs,norm_source_locations[...,None],mic_alignments[:,None,:])

        #ステアリングベクトルを算出
        steering_vector=1./np.sqrt(n_channels)*np.exp(steering_phase)

        return(steering_vector)

    else:

        #音源とマイクの距離を求める
        #distance: Ns x Nm
        distance=np.sqrt(np.sum(np.square(source_locations[...,None]-mic_alignments[:,None,:]),axis=0))

        #遅延時間(delay) [sec]
        delay=distance/sound_speed

        #ステアリングベクトルの位相を求める
        steering_phase=np.einsum('k,sm->ksm',-2.j*np.pi*freqs,delay)
    
        #音量の減衰
        steering_decay_ratio=1./distance

        #ステアリングベクトルを求める
        steering_vector=steering_decay_ratio[None,...]*np.exp(steering_phase)

        #大きさを1で正規化する
        steering_vector=steering_vector/np.linalg.norm(steering_vector,2,axis=2,keepdims=True)

    return(steering_vector)

def write_file_from_time_signal(signal,file_name,sample_rate):
    #2バイトのデータに変換
    signal=signal.astype(np.int16)

    #waveファイルに書き込む
    wave_out = wave.open(file_name, 'w')

    #モノラル:1、ステレオ:2
    wave_out.setnchannels(1)

    #サンプルサイズ2byte
    wave_out.setsampwidth(2)

    #サンプリング周波数
    wave_out.setframerate(sample_rate)

    #データを書き込み
    wave_out.writeframes(signal)

    #ファイルを閉じる
    wave_out.close()

np.random.seed(0)
# clean_wave_files = ["../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"]
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
    s = s + 1
sample_rate = 16000
N = 1024
Nk = N / 2 + 1
freqs = np.arange(0, Nk, 1) * sample_rate / N
# SNR = 20.
SNR = 90.
room_dim = np.r_[10.0, 10.0, 10.0]
mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1
# mic_alignments = np.array(
#     [[-0.01, 0.0, 0.0],
#      [ 0.01, 0.0, 0.0]]
# )
mic_alignments = np.array(
    [[x, 0.0, 0.0] for x in np.arange(-0.31, 0.32, 0.02)]
)
n_channels = np.shape(mic_alignments)[0]
R = mic_alignments.T + mic_array_loc[:, None]
room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
# doas = np.array([[np.pi / 2., 0.]])
doas = np.array([[np.pi / 2., 0.],
                 [np.pi / 2., np.pi / 2.]])
distance = 1.
source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance
source_locations += mic_array_loc[:, None]
print(np.shape(clean_data))
for s in range(n_sources):
    clean_data[s] /= np.std(clean_data[s])
    room.add_source(source_locations[:, s], signal=clean_data[s])
room.simulate(snr=SNR)
multi_conv_data = room.mic_array.signals
write_file_from_time_signal(multi_conv_data[0] * np.iinfo(np.int16).max / 20.,
                            "./results/mix_in_2spk.wav", sample_rate)
# near_steering_vectors = calculate_steering_vector(R, source_locations, freqs, is_use_far=False)
near_steering_vectors = calculate_steering_vector(R, source_locations[:,:1], freqs, is_use_far=False)
f, t, stft_data = sp.stft(multi_conv_data, fs=sample_rate, window="hann", nperseg=N)
s_hat = np.einsum("ksm,mkt->skt", np.conjugate(near_steering_vectors), stft_data)
c_hat = np.einsum("skt,ksm->mskt", s_hat, near_steering_vectors)
t, ds_out = sp.istft(c_hat[0], fs=sample_rate, window="hann", nperseg=N)
ds_out = ds_out * np.iinfo(np.int16).max / 20.
write_file_from_time_signal(ds_out, "./results/ds_out_2spk.wav", sample_rate)