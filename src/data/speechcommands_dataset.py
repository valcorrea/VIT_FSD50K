import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, 
                 root, 
                 audio_config: dict,
                 labels_map: str,
                 subset: str,
                 features: Callable = None,
                 download: bool = True,
                 duration: float = 1.0
                 ) -> None:
        super().__init__(root, "speech_commands_v0.02", "SpeechCommands", download, subset)
        with open(labels_map, 'r') as fd:
                self.labels_map = json.load(fd)
        self.features = features
        self.duration = duration

    def __getitem__(self, n: int):
        items =  super().__getitem__(n)
        audio = items[0]
        sr = items[1]
        # SpeechCommands has a fixed duration of 1.
        min_samples = int(sr * self.duration)
        if audio.shape[1] < min_samples:
            tile_size = (min_samples // audio.shape[1]) + 1
            audio = audio.repeat(1, tile_size)
            audio = audio[:, :min_samples]
    
        label_str = items[2]
        label = self.labels_map[label_str]
        if self.features is not None:
            audio = self.features(audio)

        return audio, label
