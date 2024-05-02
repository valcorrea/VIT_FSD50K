from torchaudio.datasets import SPEECHCOMMANDS
from src.data.features import LogMelSpec
import torch
import torchaudio
import matplotlib.pyplot as plt
import wandb



# Loading the SpeechCommands dataset
dataset = SPEECHCOMMANDS(root='/home/student.aau.dk/saales20/local-global-Mul-Head-Attention/VIT_FSD50K/data_set/SpeechCommands', download=True)

# Get an audio sample from the dataset
sample, sample_rate, label, *_ = dataset[0]  # Get the first sample 

log_mel_spec = LogMelSpec(sr=sample_rate, s_min=-0.1, s_max=0.1, n_time_masks=2, time_mask_width=25, n_freq_masks=5, freq_mask_width=7)

#log_mel_spec = LogMelSpec(sr=sample_rate, s_min=-0.1, s_max=0.1, r_min=0.85, r_max=1.15, n_time_masks=2, time_mask_width=25, n_freq_masks=5, freq_mask_width=7)

# Pass the audio sample through the module
mel_spec = log_mel_spec(sample)

# Visualize the log mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec.squeeze().numpy(), cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Log amplitude')
plt.xlabel('Time')
plt.ylabel('Mel frequency bin')
plt.title('Mel Spectrogram')

# Log the spectrogram image to wandb
wandb.init(project="Fourier-approximation-for-local-global-Multi-Head-Attention", entity="ce8-840")
images = wandb.Image(plt.gcf())
wandb.log({"examples": images})
wandb.finish()

