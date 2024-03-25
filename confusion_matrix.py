"""Script for training KWT model"""
from argparse import ArgumentParser
#from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from torch import nn
#import wandb
#from utils.trainer import train, evaluate
from src.data.dataset import SpectrogramDataset
from utils.config_parser import parse_config
from src.models.KWT import KWT
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

def main(args):
    """
    Calls training pipeline and sets up wandb logging if used
    :param args: input arguments
    
    """

    config = parse_config(args.conf)

    test_set = SpectrogramDataset(config['eval_manifest_path'],config['labels_map'],config['audio_config'])
    #test_loader = DataLoader(test_set, batch_size=config['hparams']['batch_size'],num_workers=5)
    #classes = ('COVID-19', 'healthy', 'symptomatic')
    classes = {"COVID-19": 0, "healthy": 1, "symptomatic": 2}
    

    test_set.files = test_set.files[:100]
    test_set.len = len(test_set.files)
    test_set.labels = test_set.labels[:100]

    print(len(test_set.files))
    #test_set.len = len(test_set.files)
    print(len(test_set))

    checkpoint = torch.load("outputs/best.pth") # loading checkpoint
    model = KWT(**config['hparams']['KWT']) #loading model
    model.load_state_dict(checkpoint['model_state_dict']) #loading state_dict
    model.eval() #evaluation on test set

    # Loading test data and creating data loader
    test_loader = DataLoader(test_set, batch_size=1) #batch size equal to testset size
    print(len(test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation loop
    predicted_labels = []
    true_labels = []

    # iterate over test data
    for spectrogram, labels in tqdm(test_loader):
        spectrogram = spectrogram.expand(1, -1, -1, -1) # Create an extra empty dimension
        spectrogram = spectrogram.permute(1, 0, 2, 3) # Permute so we have Batch - Channel - Width - Height
        output = model(spectrogram) # Feed Network
        predicted_labels.append(output.argmax(1))
        true_labels.append(labels.argmax(1))
        
    #print(predicted_labels)
    #print(true_labels)
    predicted_labels = torch.flatten(torch.tensor(predicted_labels))
    true_labels = torch.flatten(torch.tensor(true_labels))
    print(np.shape(predicted_labels))
    print(np.shape(true_labels))
    #print(predicted_labels)
    #print(true_labels)
    print(pd.Series(predicted_labels).value_counts())
    print(pd.Series(true_labels).value_counts())

    # Computing confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    #Plotting confusion matrix
    df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == "__main__":

    
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Optional experiment identifier.", default=None)
    args = parser.parse_args()

    main(args)

   