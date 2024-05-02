import json
import torch
import os
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import numpy as np
import soundfile
from glob import glob
import random

class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, 
                 root,
                 audio_config: dict,
                 labels_map: str,
                 subset: str,
                 download: bool = True,
                 duration: float = 1.0,
                 bg_prob = 0.7,
                 normalize=True,
                 flip_ft=False,
                 r_min=0.85,
                 r_max=1.15,
                 s_min=-0.1,
                 s_max=0.1,
                 num_frames=100,
                 n_time_masks=2,
                 time_mask_width=25,
                 n_freq_masks=5,
                 freq_mask_width=7,
                 seed=42
                 ) -> None:
        super().__init__(root, "speech_commands_v0.02", "SpeechCommands", download, subset)
        with open(labels_map, 'r') as fd:
                self.labels_map = json.load(fd)
        
        random.seed(seed)
        np.random.seed(seed)

        self.duration = duration
        self.r_max = r_max
        self.r_min = r_min

        # Background
        self.bg_files = glob(os.path.join(root, "SpeechCommands", "speech_commands_v0.02", "_background_noise_", '*.wav'))
        self.bg_prob = bg_prob

        # Time-shift Config
        self.s_min = s_min
        self.s_max = s_max

        # Masking Config
        self.n_time_masks = n_time_masks
        self.time_mask_width = time_mask_width
        self.n_freq_masks = n_freq_masks
        self.freq_mask_width = freq_mask_width

        # Audio Config
        self.sr = audio_config.get('sr', 16000)
        self.n_fft = audio_config.get('n_fft', 400)
        self.win_len = audio_config.get('win_len', 400)
        self.hop_len = audio_config.get('hop_len', 160)
        self.f_min = audio_config.get('f_min', 50)
        self.f_max = audio_config.get('f_max', 8000)
        self.n_mels = audio_config.get('n_mels', 80)

        self.normalize = normalize
        self.flip_ft = flip_ft
    
        if num_frames is not None:
            self.num_frames = int(num_frames)
        else:
            self.num_frames = None

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.n_fft, win_length=self.win_len, hop_length=self.hop_len,
            f_min=self.f_min, f_max=self.f_max,
            n_mels=self.n_mels, power=2.
        )

    def resample(self, x):

        r_min = self.r_min
        r_max = self.r_max
        sr_new = int(self.sr * np.random.uniform(r_min, r_max))
        x_resampled = torchaudio.transforms.Resample(self.sr, sr_new)(x)
        return x_resampled
    
    def time_shift(self, x):
        sr = self.sr
        s_min = self.s_min
        s_max = self.s_max
        start = int(np.random.uniform(sr * s_min, sr * s_max))
        if start >= 0:
            shifted_x = torch.cat((x[:, start:], torch.FloatTensor(np.random.uniform(-0.001, 0.001, start)).unsqueeze(0)), dim=1)
        else:
            shifted_x = torch.cat((torch.FloatTensor(np.random.uniform(-0.001, 0.001, -start)).unsqueeze(0), x[:, :start]), dim=1)
        return shifted_x
    
    def spec_augment(self, mel_spec):
        n_time_masks = self.n_time_masks
        time_mask_width = self.time_mask_width
        n_freq_masks = self.n_freq_masks
        freq_mask_width = self.freq_mask_width

        mel_spec_copy = mel_spec.clone()  # Make a copy of the input mel_spec

        for _ in range(n_time_masks):
            offset = np.random.randint(0, time_mask_width)
            begin = np.random.randint(0, mel_spec_copy.shape[2] - offset)
            mel_spec_copy[:, :, begin: begin + offset] = 0.0
        
        for _ in range(n_freq_masks):
            offset = np.random.randint(0, freq_mask_width)
            begin = np.random.randint(0, mel_spec_copy.shape[1] - offset)
            mel_spec_copy[:, begin: begin + offset, :] = 0.0

        return mel_spec_copy


    def add_background(self, x):
        bg_path = random.choice(self.bg_files)
        bg_wav = soundfile.read(bg_path)[0]
        index = random.randint(0, len(bg_wav-len(x)))
        background_cropped = torch.tensor(bg_wav[index:index+len(x)])
        wav_with_bg = x.add(background_cropped).float()
        return wav_with_bg


    def __getitem__(self, n: int):
        # Get waveform from speech commands
        items =  super().__getitem__(n)
        audio = items[0]
        sr = items[1]
        
        # Add background
        if random.random() < self.bg_prob:
            audio = self.add_background(audio)

        #resampling audio sample
        audio = self.resample(audio)

        # SpeechCommands has a fixed duration of 1.
        min_samples = int(sr * self.duration)
        if audio.shape[1] < min_samples:
            tile_size = (min_samples // audio.shape[1]) + 1
            audio = audio.repeat(1, tile_size)
            audio = audio[:, :min_samples]
    
        label_str = items[2]
        label = self.labels_map[label_str]

        # Applying time shifting
        audio = self.time_shift(audio)

        # Computing log mel spectrogram
        audio = self.melspec(audio)
        audio = (audio + torch.finfo().eps).log()

        if self.num_frames is not None:
            audio = audio[:, :, :self.num_frames]
    
        if self.normalize:
            mean = torch.mean(audio, [1, 2], keepdims=True)
            std = torch.std(audio, [1, 2], keepdims=True)
            audio = (audio - mean) / (std + 1e-8)
        
        if self.flip_ft:
            audio = audio.transpose(-2, -1)

        # Applying SpecAug
        audio = self.spec_augment(audio)
        return audio, label
    
    
