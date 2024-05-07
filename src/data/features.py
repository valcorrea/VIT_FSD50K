import torch
import torch.nn as nn
import torchaudio
import numpy as np

class LogMelSpec(nn.Module):
    def __init__(
        self, 
        sr=16000,
        n_mels=80,
        n_fft=400,
        win_len=400,
        hop_len=160,
        f_min=50.,
        f_max=8000.,
        normalize=True,
        flip_ft=False,
        num_frames=None,
        s_min=-0.1,
        s_max=0.1,
        n_time_masks=2,
        time_mask_width=25,
        n_freq_masks=5,
        freq_mask_width=7
    ) -> None:
        super().__init__()
        self.sr = sr
        self.s_min = s_min
        self.s_max = s_max
        self.n_time_masks = n_time_masks
        self.time_mask_width = time_mask_width
        self.n_freq_masks = n_freq_masks
        self.freq_mask_width = freq_mask_width

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=win_len, hop_length=hop_len,
            f_min=f_min, f_max=f_max,
            n_mels=n_mels, power=2.
        )
        self.normalize = normalize
        self.flip_ft = flip_ft
        if num_frames is not None:
            self.num_frames = int(num_frames)
        else:
            self.num_frames = None

    def forward(self, x):
        # Applying time shifting
        shifted_x = self.time_shift(x)

        # Computing log mel spectrogram
        x = self.melspec(shifted_x)
        x = (x + torch.finfo().eps).log()

        if self.num_frames is not None:
            x = x[:, :, :self.num_frames]
    
        if self.normalize:
            mean = torch.mean(x, [1, 2], keepdims=True)
            std = torch.std(x, [1, 2], keepdims=True)
            x = (x - mean) / (std + 1e-8)
        
        if self.flip_ft:
            x = x.transpose(-2, -1)

        # Applying SpecAug
        x = self.spec_augment(x)
        
        return x

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

        print("Original mel_spec shape:", mel_spec.shape)

        mel_spec_copy = mel_spec.clone()  # Make a copy of the input mel_spec

        for _ in range(n_time_masks):
            offset = np.random.randint(0, time_mask_width)
            begin = np.random.randint(0, mel_spec_copy.shape[2] - offset)
            print("Time mask:", begin, begin + offset)
            mel_spec_copy[:, :, begin: begin + offset] = 0.0
        
        for _ in range(n_freq_masks):
            offset = np.random.randint(0, freq_mask_width)
            begin = np.random.randint(0, mel_spec_copy.shape[1] - offset)
            print("Freq mask:", begin, begin + offset)
            mel_spec_copy[:, begin: begin + offset, :] = 0.0

        print("Final mel_spec shape:", mel_spec_copy.shape)

        return mel_spec_copy

# Example usage:
# log_mel_spec = LogMelSpec()
# mel_spec = log_mel_spec(audio_waveform)
