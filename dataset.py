import glob
import json
import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset

df = pd.read_csv("../../covid_dataset/archive/metadata_compiled.csv")

# cough_detected threshold = 0.8  and valid status labels, so it filters the empty samples
valid_samples = df[(df["cough_detected"] > 0.8) & df["status"].notnull()]

audio_files = valid_samples["uuid"].values

# Label ['healthy' 'symptomatic' 'COVID-19']     np.unique(labels)
labels = valid_samples["status"].values

# Make sure the audio files and labels have the same length
assert len(audio_files) == len(labels)

length = len(audio_files)

# Audio dictionary
sample_rate = 22050
n_fft = 511  # This determines the frequency resolution of the spectrogram
hop_len = 220  # This determines the time resolution of the spectrogram
normalize = True
min_duration = None  # Specifies the minimum duration of the audio data in seconds
feature = "spectrogram"  # melspectrogram

# if no window_len in the dictionary
window_len = n_fft
# and if no hop_len
hop_len = n_fft // 2

augment = False
mode = "multilabel"  # "multiclass"


class SpectrogramDataset(Dataset):
    def __init__(self) -> None:
        super(SpectrogramDataset, self).__init__()
        pd.read_csv()
