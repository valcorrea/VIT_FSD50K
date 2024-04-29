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
                 ) -> None:
        super().__init__(root, "speech_commands_v0.02", "SpeechCommands", download, subset)
        with open(labels_map, 'r') as fd:
                self.labels_map = json.load(fd)
        self.features = features

    def __getitem__(self, n: int):
        items =  super().__getitem__(n)
        audio = items[0]
        label_str = items[2]
        label = self.labels_map[label_str]
        if self.features is not None:
            audio = self.features(audio)

        return audio, label
