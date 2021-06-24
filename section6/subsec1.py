import numpy as np

def calculate_steering_vector(mic_alignments, source_locations, freqs,
                              sound_speed=340, is_use_far=False):
    n_channels = np.shape(mic_alignments)[1]
    n_sources = np.shape(source_locations)[1]
    if is_use_far == True:
        norm_source_locations = source_locations / np.linalg.norm(
            source_locations, 2, axis=0, keepdims=True)
        steering_phase = np.einsum('k,ism,ism->ksm', 2.j * np.pi / sound_speed * freqs,
            norm_source_locations[...,None], mic_alignments[:,None,:])
        steering_vector = 1. / np.sqrt(n_channels) * np.exp(steering_phase)
        return steering_vector
    else:
        distance = np.sqrt(np.sum(np.square(source_locations[...,None] - mic_alignments[:,None,:]), axis=0))
        delay = distance / sound_speed
        steering_phase = np.einsum('k,sm->ksm', -2.j * np.pi * freqs, delay)
        steering_decay_ratio = 1. / distance
        steering_vector = steering_decay_ratio[None,...] * np.exp(steering_phase)
        steering_vector = steering_vector / np.linalg.norm(steering_vector, 2, axis=2, keepdims=True)
        return steering_vector

sample_rate = 16000
N = 1024
Nk = N / 2 + 1
freqs = np.arange(0, Nk, 1) * sample_rate / N
mic_alignments = np.array(
    [[-0.01, 0.0, 0.0],
     [0.01, 0.0, 0.0],]
).T

doas = np.array([[np.pi / 2., 0],
                 [np.pi / 2., np.pi]])
distance = 1.
source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance

near_steering_vectors = calculate_steering_vector(mic_alignments, source_locations, freqs, is_use_far=False)
far_steering_vectors = calculate_steering_vector(mic_alignments, source_locations, freqs, is_use_far=True)
inner_product = np.einsum("ksm,ksm->ks", np.conjugate(near_steering_vectors), far_steering_vectors)
print(np.average(np.abs(inner_product)))