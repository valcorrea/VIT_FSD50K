import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
import lightning as L
from src.models.ssformer_lightning import LightningTransformer

from src.utils.config_parser import parse_config
from src.utils.ssformer_trainer import train
from src.utils.masking import AudioMaskingGenerator
from src.models.KWT import KWT
from src.models.ssformer import SSTransformer

from src.data.dataset import SpectrogramDataset



def training_pipeline(config):

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize KWT
    vit = KWT(**config['hparams']['KWT'])
    vit.to(device);

    # Initialize SSformer
    ssformer = LightningTransformer(encoder=vit, config=config)
    ssformer.to(device)

    # Set criterion and optimizer
    # criterion = nn.MSELoss(reduction="none")
    # parameters = ssformer.parameters()
    # optimizer = optim.Adam(parameters, lr=config["hparams"]["optimizer"]["lr"],
    #                        betas=config["hparams"]["optimizer"]["betas"],
    #                        eps=config["hparams"]["optimizer"]["eps"],
    #                        weight_decay=config["hparams"]["optimizer"]["weight_decay"])

    # Make dataset
    train_set = SpectrogramDataset(config['nl_manifest_path'], labels_map=None, mode=None, audio_config=config['audio_config'])
    val_set = SpectrogramDataset(config['nl_manifest_path'], labels_map=None, mode=None, audio_config=config['audio_config'])

    # Make dataloaders
    train_loader = DataLoader(train_set, batch_size=config['hparams']['batch_size'])
    val_loader = DataLoader(val_set, batch_size=config['hparams']['batch_size'])

    # Make a mask generator???
    mask_generator = AudioMaskingGenerator(mask_prob=config["hparams"]["SSformer"]["mask_prob"],
                                           mask_length=config["hparams"]["SSformer"]["mask_length"],
                                           attention_mask=None,
                                           min_masks=config["hparams"]["SSformer"]["min_masks"])

    # Make a scheduler???
    schedulers = {'scheduler': None,
                  'warmup': None}

    trainer = L.Trainer()
    trainer.fit(ssformer, train_loader, val_loader)
    # Train
    # train(net=ssformer,
    #       mask_generator=mask_generator,
    #       optimizer=optimizer,
    #       criterion=criterion,
    #       train_loader=train_loader,
    #       validation_loader=val_loader,
    #       schedulers=schedulers,
    #       config=config)
    
    wandb.finish()
    

def main(args):

    config = parse_config(args.conf)
    config['nl_manifest_path'] = args.nl_manifest_path
    
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id
    
    if config["exp"]["wandb"]:
        
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()

        elif os.environ.get("WANDB_API_KEY", False):
            print(f"Found API key from env variable.")

        else:
            wandb.login()

        with wandb.init(project=config["exp"]["proj_name"], 
                        name=config["exp"]["exp_name"],
                        entity=config["exp"]["entity"], 
                        config=config["hparams"]):
            training_pipeline(config)

    else:
        training_pipeline(config)



if __name__ == '__main__':
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('--conf', type=str, required=True, help='Path to configuration file')
    ap.add_argument('--id', type=str, help='Unique experiment identifier')
    ap.add_argument('--nl_manifest_path', type=str, help='Path to the unlabeled data manifest.')
    args = ap.parse_args()

    main(args)

