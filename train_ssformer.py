'''
Script to train a Self-Supervised ViT
'''

import os
import wandb

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from utils.light_modules import LightningSSformer
from utils.config_parser import parse_config
from src.models.KWT import KWT
from src.data.dataset import SpectrogramDataset

def training_pipeline(config, logger, ckpt_path):

    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # Initialize KWT
    vit = KWT(**config['hparams']['KWT'])
    vit.to(device);

    # Initialize SSformer
    if ckpt_path:
        ssformer = LightningSSformer.load_from_checkpoint(ckpt_path, encoder=vit, config=config)
    else:
        ssformer = LightningSSformer(encoder=vit, config=config)
    ssformer.to(device)

    # Make dataset
    train_set = SpectrogramDataset(config['tr_manifest_path'], labels_map=None, mode=None, audio_config=config['audio_config'])
    val_set = SpectrogramDataset(config['val_manifest_path'], labels_map=None, mode=None, audio_config=config['audio_config'])

    if config['dev_mode']:
        train_set.files = train_set.files[:50]
        train_set.len = len(train_set.files)
        val_set.files = val_set.files[:50]
        val_set.len = len(val_set.files)
        config['hparams']['batch_size'] = 25


    # Make dataloaders
    train_loader = DataLoader(train_set, batch_size=config['hparams']['batch_size'], num_workers=5)
    val_loader = DataLoader(val_set, batch_size=config['hparams']['batch_size'], num_workers=5)

    # Create Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", verbose=True)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=config['hparams']['early_stopping_patience'], verbose=True)
    callbacks = [model_checkpoint, early_stopping]

    trainer = L.Trainer(max_epochs=config['hparams']['n_epochs'], 
                        logger=logger,
                        callbacks=callbacks,
                        log_every_n_steps=5,
                        strategy='ddp_find_unused_parameters_true')
    trainer.fit(ssformer, train_loader, val_loader) 

def main(args):

    config = parse_config(args.config)
    if args.tr_manifest_path:
        config['tr_manifest_path'] = args.tr_manifest_path
    if args.val_manifest_path:
        config['val_manifest_path'] = args.val_manifest_path
    config['dev_mode'] = args.dev_mode
    
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
            logger = WandbLogger(project=config["exp"]["proj_name"],
                                 name=config["exp"]["exp_name"],
                                 entity=config["exp"]["entity"],
                                 config=config["hparams"],
                                 log_model=True,
                                 save_dir='outputs/')
            training_pipeline(config, logger, args.ckpt_path)

    else:
        logger = None
        training_pipeline(config, logger, args.ckpt_path)

if __name__ == '__main__':
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('config', type=str, help='Path to configuration file')
    ap.add_argument('--id', type=str, help='Unique experiment identifier')
    ap.add_argument('--tr_manifest_path', type=str, help='Path to the unlabeled train data manifest.')
    ap.add_argument('--val_manifest_path', type=str, help='Path to the unlabeled val data manifest.')
    ap.add_argument('--ckpt_path', type=str, help='Path to model checkpoint.')
    ap.add_argument('--dev_mode', action='store_true', help='Flag to limit the dataset for testing purposes.')
    args = ap.parse_args()

    main(args)

