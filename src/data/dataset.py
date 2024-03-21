# Taken from (https://github.com/SarthakYadav)
# MIT License
# Copyright (c) 2021 Sarthak Yadav


"""
Spectrogram Dataset modified it to be able to work with Covid-19 Cough Audio Classification dataset(https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification)

"""




import glob
import json
import os
from typing import Optional, Tuple

import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset

from src.data.audio_parser import AudioParser
from src.data.utils import load_audio


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        labels_map: str,
        audio_config: dict,
        mode: Optional[str] = "multilabel",
        augment: Optional[bool] = False,
        labels_delimiter: Optional[str] = ",",
        mixer=None,
        transform=None,
        extra_labels: Optional[str] = None,
    ) -> None:
        super(SpectrogramDataset, self).__init__()

        assert audio_config is not None
        df = pd.read_csv(manifest_path)
        self.files = df["files"].values 
        # Set labels_map to None for self supervised.

        #self.labels is populated with the labels extracted from the CSV file. 
        #If labels_map is provided, it reads the labels from the CSV file and maps them according to the labels_map
        if labels_map:
            assert os.path.isfile(labels_map)
            assert os.path.splitext(labels_map)[-1] == ".json"
            self.labels_delim = labels_delimiter
            with open(labels_map, "r") as fd:
                self.labels_map = json.load(fd)
            self.labels = df["labels"].values
            assert len(self.files) == len(self.labels)
        else:
            self.labels = None

        self.len = len(self.files)
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        self.extra_labels = []
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = win_len
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = hop_len

        self.normalize = audio_config.get("normalize", True)
        self.augment = augment
        self.min_duration = audio_config.get("min_duration", None)
        self.background_noise_path = audio_config.get("bg_files", None)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(
                    os.path.join(self.background_noise_path, "*.wav")
                )
        else:
            self.bg_files = None

        self.mode = mode
        feature = audio_config.get("feature", "spectrogram")
        self.spec_parser = AudioParser(
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
            feature=feature,
        )
        self.mixer = mixer
        self.transform = transform

        if self.bg_files is not None:
            print("prepping bg_features")
            self.bg_features = []
            for f in tqdm.tqdm(self.bg_files):
                preprocessed_audio = self.__get_audio__(f)
                real, comp = self.__get_feature__(preprocessed_audio)
                self.bg_features.append(real)
        else:
            self.bg_features = None
        self.prefetched_labels = None
        if self.mode == "multilabel":
            self.fetch_labels()

        if extra_labels != None:
            extra_lbl = pd.read_csv(extra_labels)
            self.extra_labels[extra_labels["age"]]

        if labels_map:
            self.weighted_sampler = self.calc_weights()

    def fetch_labels(self):
        self.prefetched_labels = []
        for lbl in tqdm.tqdm(self.labels):
            proc_lbl = self.__parse_labels__(lbl)
            self.prefetched_labels.append(proc_lbl.unsqueeze(0))
        self.prefetched_labels = torch.cat(self.prefetched_labels, dim=0)
        print(self.prefetched_labels.shape)

    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def get_bg_feature(self, index: int) -> torch.Tensor:
        if self.bg_files is None:
            return None
        real = self.bg_features[index]
        if self.transform is not None:
            real = self.transform(real)
        return real

    def __get_item_helper__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.files[index]
        preprocessed_audio = self.__get_audio__(f)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.transform is not None:
            real = self.transform(real)
        #if len(self.labels) > 0:
        if self.labels is not None:
            lbls = self.labels[index]
            label_tensor = self.__parse_labels__(lbls)
            return real, comp, label_tensor
        else:
            return real, comp

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Fix mixer for self supervised
        #if len(self.labels) > 0:
        if self.labels is not None:    
            real, comp, label_tensor = self.__get_item_helper__(index)
            if self.mixer is not None:
                real, final_label = self.mixer(self, real, label_tensor)
                if self.mode != "multiclass":

                    return real, final_label

            return real, label_tensor
        else:
            real, comp = self.__get_item_helper__(index)
            return real

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        if self.mode == "multilabel":
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == "multiclass":
            return self.labels_map[lbls]

    def __len__(self):
        return self.len

    def get_bg_len(self):
        return len(self.bg_files)
    
    # Weight calculation
    #def calc_weights(self):

        
    #    from torch.utils.data import WeightedRandomSampler
    #    counts = self.annotations['label'].value_counts() #replace self.annotations with equivalent
    #    weights = []
    #    for row in self.annotations.iterrows():
    #        weights.append(1./counts[row[1]['label']])

    #    weights = torch.FloatTensor(weights)
    #    sampler = WeightedRandomSampler(weights, len(weights))
    #    return sampler
    

    def calc_weights(self):
        from torch.utils.data import WeightedRandomSampler
        # Initialize an empty array for weights
        weights = []
        
        # Calculate label counts
        label_counts = {}
        for labels in self.labels:
            for label in labels.split(self.labels_delim):
                label_counts[label] = label_counts.get(label, 0) + 1
                
        for labels in self.labels:
            weight = sum(1.0 / label_counts[label] for label in labels.split(self.labels_delim))
            weights.append(weight)
        
        weights = torch.FloatTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler
            
    
if __name__ == '__main__':
    from src.utils.config_parser import parse_config
    from torch.utils.data import DataLoader
    config = parse_config('configs/ssformer.cfg')
    manifest_path = r"C:\Users\vismi\Documents\datasets\covid19-cough\manidests\no_labeled_chunk.csv"

    dataset = SpectrogramDataset(manifest_path = manifest_path,
                                 labels_map= None,
                                audio_config = config['audio_config'],
                                mode = "multiclass")
    
    dataloader = DataLoader(dataset, batch_size=1)
    print(next(iter(dataloader[0].shape)))