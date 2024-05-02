import numpy as np
import numba as nb
import librosa

@nb.jit(nopython=True, cache=True)
def time_shift(wav: np.ndarray, sr: int, s_min: float, s_max: float) -> np.ndarray:
    """Time shift augmentation.
    Refer to https://www.kaggle.com/haqishen/augmentation-methods-for-audio#1.-Time-shifting.
    Changed np.r_ to np.hstack for numba support.

    Args:
        wav (np.ndarray): Waveform array of shape (n_samples,).
        sr (int): Sampling rate.
        s_min (float): Minimum fraction of a second by which to shift.
        s_max (float): Maximum fraction of a second by which to shift.
    
    Returns:
        wav_time_shift (np.ndarray): Time-shifted waveform array.
    """

    start = int(np.random.uniform(sr * s_min, sr * s_max))
    if start >= 0:
        wav_time_shift = np.hstack((wav[start:], np.random.uniform(-0.001, 0.001, start)))
    else:
        wav_time_shift = np.hstack((np.random.uniform(-0.001, 0.001, -start), wav[:start]))
    
    return wav_time_shift

def resample(x: np.ndarray, sr: int, r_min: float, r_max: float) -> np.ndarray:
    """Resamples waveform.

    Args:
        x (np.ndarray): Input waveform, array of shape (n_samples, ).
        sr (int): Sampling rate.
        r_min (float): Minimum percentage of resampling.
        r_max (float): Maximum percentage of resampling.
    """

    sr_new = sr * np.random.uniform(r_min, r_max)
    x = librosa.resample(x, sr, sr_new)
    return x, sr_new
  

@nb.jit(nopython=True, cache=True)
def spec_augment(mel_spec: np.ndarray, n_time_masks: int, time_mask_width: int, n_freq_masks: int, freq_mask_width: int):
    """Numpy implementation of spectral augmentation.

    Args:
        mel_spec (np.ndarray): Mel spectrogram, array of shape (n_mels, T).
        n_time_masks (int): Number of time bands.   
        time_mask_width (int): Max width of each time band.
        n_freq_masks (int): Number of frequency bands.
        freq_mask_width (int): Max width of each frequency band.

    Returns:
        mel_spec (np.ndarray): Spectrogram with random time bands and freq bands masked out.
    """
    
    offset, begin = 0, 0

    for _ in range(n_time_masks):
        offset = np.random.randint(0, time_mask_width)
        begin = np.random.randint(0, mel_spec.shape[1] - offset)
        mel_spec[:, begin: begin + offset] = 0.0
    
    for _ in range(n_freq_masks):
        offset = np.random.randint(0, freq_mask_width)
        begin = np.random.randint(0, mel_spec.shape[0] - offset)
        mel_spec[begin: begin + offset, :] = 0.0

    return mel_spec



########### Parameters for time shift according to KWT paper ###################
s_min = -0.1  # Minimum time shift in seconds (-100 ms)
s_max = 0.1   # Maximum time shift in seconds (100 ms)
sr = 16000 # Sample rate, 16.0 kHz

########### Parameters for resampling according to KWT paper ###################
r_min = 0.85
r_max = 1.15


########### Parameters for SpecAugmentation according to KWT paper ###################
n_time_masks = 2
time_mask_width = 25
n_freq_masks = 5
freq_mask_width = 7

#resampled_wav, new_sr = resample(wav, sr, r_min, r_max)
#shifted_wav = time_shift(wav, sr, s_min, s_max)
#spec_augment = spec_augment(mel_spec, n_time_masks, time_mask_width, n_freq_masks, freq_mask_width)

