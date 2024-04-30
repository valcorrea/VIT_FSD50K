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

def test_pipeline(model,testing_loader,device):
    # model.eval()
    correct_predictions = 0
    total_samples = len(testing_loader.dataset)
    print(f"Total_samples {total_samples}")
    print(f"Num_batches {len(testing_loader)}")
    
    trainer = L.Trainer()

    trainer.fit(model, testing_loader)
    trainer.test()
    
    # for ix in tqdm.tqdm(range(testing_loader.len)):
    #     with torch.no_grad():
    #         x, y = testing_loader[ix]
    #         x = x.to(device)
    #         y_pred = model(x)
            
    # with torch.no_grad():
    #     for batch in testing_loader:
    #         inputs, labels = batch
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         print(f"Inputs shape:{inputs.shape}")
    #         # Forward pass
    #         outputs = model(inputs)

    #         # Get predicted labels
    #         _, predicted = torch.max(outputs, 1)

    #         # Count correct predictions
    #         correct_predictions += (predicted == labels).sum().item()

    # # Calculate accuracy
    # accuracy = correct_predictions / total_samples * 100
    # print(f"Testing Accuracy: {accuracy:.2f}%")
    
def test_pipeline_2(model, test_set, device, classes):
    print("Initiating testing...")

    predicted_labels, true_labels = [], []

    for ix in tqdm.tqdm(range(test_set.len)):
        with torch.no_grad():
            x, y = test_set[ix]
            x = x.to(device)
            y_pred = model(x)
            y_pred = y_pred.mean(0).unsqueeze(0)
            sigmoid_preds = torch.sigmoid(y_pred)

            predicted_labels.append(sigmoid_preds.detach().cpu().numpy()[0])
            true_labels.append(y.detach().cpu().numpy()[0])

    predicted_labels = np.asarray(predicted_labels).astype('float32')
    true_labels = np.asarray(true_labels).astype('int32')
    
    #print("Shape of predicted_labels:", predicted_labels.shape)
    #print("Shape of true_labels:", true_labels.shape)
    #print("Shape of true_labels:", true_labels.shape)

    stats = calculate_stats(predicted_labels, true_labels)
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(mAP))
    print("mAUC: {:.6f}".format(mAUC))
    print("dprime: {:.6f}".format(d_prime(mAUC)))

    # Plotting mAP
   # plt.figure(figsize=(6, 4))
    #plt.bar(['mAP'], [mAP[0]], color='blue')
    #plt.ylabel('Value')
    #plt.title('Mean Average Precision (mAP)')
    #plt.show()

    # Plotting mAUC
    #plt.figure(figsize=(6, 4))
    #plt.bar(['mAUC'], [mAUC[0]], color='orange')
    #plt.ylabel('Value')
    #plt.title('Mean Area Under Curve (mAUC)')
    #plt.show() 

def main(args):
    config = parse_config(args.conf)
    model, device = get_model(args.ckpt, config)
    test_set,test_loader = get_data(config)
    
     # Print the shape of the first spectrogram in the training set
    spectrogram, _ = next(iter(test_loader))
    print("Shape of the first spectrogram in the training set:", spectrogram.shape)

    # Logging setup
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id

    if config["exp"]["wandb"]:
        wandb.login()
        with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"],entity=config["exp"]["entity"]):
            test_pipeline(model, test_set, device, test_loader)

    else:
        test_pipeline(model, test_set, device,test_loader)
    
if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Optional experiment identifier.", default=None) 
    args = parser.parse_args()

    main(args)


