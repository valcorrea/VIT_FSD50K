import os
import tqdm
import numpy as np
import torch
from utils.metrics_helper import calculate_stats, d_prime
from utils.config_parser import parse_config
from argparse import ArgumentParser
import wandb
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from matplotlib import pyplot as plt
from src.data.fsd50k_dataset import SpectrogramDataset
from utils.light_modules import LightningKWT
from src.data.test_fsd50k_dataset import FSD50kEvalDataset, _collate_fn_eval
import lightning as L
from src.data.speechcommands_dataset import SpeechCommands
from src.data.features import LogMelSpec
from torch.utils.data import DataLoader

def get_model(ckpt, config, useFNet=False):
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    if ckpt:
        print("Loading from checkpoint")
        model = LightningKWT.load_from_checkpoint(ckpt, config=config)
    elif useFNet:
        model = LightningKWT(config, True)
    else:
        model = LightningKWT(config)
    model.to(device)
    print(device)
    
    return model, device


def get_data(config):
    features = LogMelSpec(
        sr=config['audio_config']['sample_rate'],
        n_mels=config['audio_config']['n_mels'],
        num_frames=config['audio_config'].get('num_frames', 100)
    )
    eval_set = SpeechCommands(root=config['dataset_root'], 
                             audio_config=config['audio_config'], 
                             labels_map=config['labels_map'], 
                             subset='testing', features=features)
    
    val_loader = DataLoader(eval_set, batch_size=config['hparams']['batch_size'], num_workers=5)
    
#     # creating transform from NumPy array to Tensor
#     transform = torch.tensor
    
#    # test_set = SpectrogramDataset(manifest_path=config['eval_manifest_path'], labels_map=config['labels_map'], audio_config=config['audio_config'])
#     test_set = FSD50kEvalDataset(manifest_path=config['eval_manifest_path'], labels_map=config['labels_map'], audio_config=config['audio_config'], transform=transform)


    # For testing purposes, use a smaller subset of the test set
    #test_set.files = test_set.files[:100]
    #test_set.labels = test_set.labels[:100]
    #test_set.len = 100
    
    
    classes = {"backward": 0, "bed": 1, "bird": 2, "cat": 3, "dog": 4, "down": 5, "eight": 6, "five": 7, "follow": 8, "forward": 9, "four": 10, "go": 11, "happy": 12, "house": 13, "learn": 14, "left": 15, "marvin": 16, "nine": 17, "no": 18, "off": 19, "on": 20, "one": 21, "right": 22, "seven": 23, "sheila": 24, "six": 25, "stop": 26, "three": 27, "tree": 28, "two": 29, "up": 30, "visual": 31, "wow": 32, "yes": 33, "zero": 34}
    return val_loader

def get_predictions(device, model, test_loader):
    predicted_labels = torch.tensor([]).to(device)
    true_labels = torch.tensor([]).to(device)

    for specs, labels in tqdm(test_loader):
        specs = specs.to(device)  #sending spectograms to device
        labels = labels.to(device) # sending labels to device
        output = model(specs)  # feeding data to network
        predicted_labels = torch.cat((predicted_labels, output.argmax(1))) #argmax to find "hot one" in one hot encoding
        true_labels = torch.cat((true_labels, labels.argmax(1))) #argmax to find "hot one" in one hot encoding
        
    predicted_labels = torch.flatten(predicted_labels).cpu() #flattening dimensions
    true_labels = torch.flatten(true_labels).cpu() #flattening dimensions
    return predicted_labels, true_labels

 
def test_pipeline(model, test_loader, device, classes):
    print("Initiating testing...")
    model.eval()
    predicted_labels, true_labels = get_predictions(device, model, test_loader)

    
    print("Shape of predicted_labels:", predicted_labels.shape)
    print("Shape of true_labels:", true_labels.shape)
    

def main(args):
    config = parse_config(args.conf)
    model, device = get_model(args.ckpt, config)
    test_loader= get_data(config)
    
     # Print the shape of the first spectrogram in the training set
    spectrogram, _ = next(iter(test_loader))
    print("Shape of the first spectrogram in the training set:", spectrogram.shape)

    # Logging setup
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id

    if config["exp"]["wandb"]:
        wandb.login()
        with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"],entity=config["exp"]["entity"]):
            test_pipeline(model, test_loader, device)

    else:
        test_pipeline(model, test_loader, device)
    
if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Optional experiment identifier.", default=None) 
    args = parser.parse_args()

    main(args)

# --conf configs/small_ViT_train_config_speech_commands_vcb.cfg --ckpt model.ckpt
