import wave as wave
import pyroomacoustics as pa
import numpy as np

pa.datasets.CMUArcticCorpus(basedir="../CMU_ARCTIC", download=True, speaker=["aew", "axb"])
sample_wave_file="../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"

wav = wave.open(sample_wave_file)