'''
Hyperparameter Optimization

About wandb sweeps:
    The sweep is its own 'construct' in wandb. It accepts the entity and project name only. 
    It then creates 'child runs' that appear under the normal runs interface.
    When defining a sweep you need to specify hyperparameters in the following format:
    parameters:
        parameter_name:
            values: list_of_values
                    OR
            distribution: distribution_name
            min:
            max:
    Anything that does not follow this specific format cannot be passed into the config file.
'''
from functools import partial
import wandb

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from utils.light_modules import LightningKWT
from utils.config_parser import parse_config
from src.data.fsd50k_dataset import SpectrogramDataset

def main(args):
    config = parse_config(args.config)
    config['dev_mode'] = args.dev_mode
    sweep_id = wandb.sweep(config['sweep'], entity=config['exp']['entity'], project=config['exp']['proj_name'])
    wandb.agent(sweep_id, partial(train, model_config=config, ckpt=args.ckpt_path), count=10)
    

def train(config=None, model_config=None, ckpt=None):
    with wandb.init(config=config):
        sweep_config = wandb.config
        train_loader, val_loader = get_dataloaders(sweep_config, model_config)
        model = get_model(ckpt, sweep_config, model_config)
        model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", verbose=True)
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=model_config['hparams']['early_stopping_patience'], verbose=True)
        callbacks = [model_checkpoint, early_stopping]

        logger = WandbLogger()

        trainer = L.Trainer(max_epochs=sweep_config['n_epochs'],
                            logger=logger,
                            callbacks=callbacks,
                            log_every_n_steps=100,
                            default_root_dir=model_config['exp']['save_dir'])
        trainer.fit(model, train_loader, val_loader)

def get_dataloaders(sweep_config, model_config):
    # Make datasets
    train_set = SpectrogramDataset(manifest_path=model_config['tr_manifest_path'], labels_map=model_config['labels_map'], audio_config=model_config['audio_config'])
    val_set = SpectrogramDataset(manifest_path=model_config['val_manifest_path'], labels_map=model_config['labels_map'], audio_config=model_config['audio_config'])
    
    if model_config['dev_mode']:
        train_set.files = train_set.files[:50]
        train_set.len = len(train_set.files)
        val_set.files = val_set.files[:50]
        val_set.len = len(val_set.files)

    # Make dataloaders
    train_loader = DataLoader(train_set, batch_size=sweep_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=sweep_config.batch_size)
    return train_loader, val_loader


def get_model(ckpt, sweep_config, model_config):
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
        
    if ckpt:
        print('Loading from checkpoint')
        model = LightningKWT.load_from_checkpoint(ckpt, hparams=sweep_config, arch=model_config['hparams']['KWT'])
    else:
        model = LightningKWT(hparams=sweep_config, arch=model_config['hparams']['KWT'])
    model.to(device)
    return model


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--config', type=str, help='Path to the model configuration file.')
    ap.add_argument('--ckpt_path', type=str, help='Path to model checkpoint.')
    ap.add_argument('--dev_mode', action='store_true', help='Flag to limit the dataset for testing purposes.')
    args = ap.parse_args()

    main(args)