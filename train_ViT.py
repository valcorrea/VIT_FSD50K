
import wandb

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import torch
from torch.utils.data import DataLoader

from utils.config_parser import parse_config
from utils.light_modules import LightningViT
from src.data.fsd50k_dataset import SpectrogramDataset


def main(args):

    config = parse_config(args.config)
    config['dev_mode'] = args.dev_mode
    config['preload_data'] = args.preload_data
    
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id
    
    if config["exp"]["wandb"]:
        wandb.login()
        logger = WandbLogger(project=config["exp"]["proj_name"],
                                name=config["exp"]["exp_name"],
                                entity=config["exp"]["entity"],
                                config=config["hparams"],
                                log_model=True,
                                save_dir=config["exp"]["save_dir"])
    else:
        logger = None
    
    model = get_model(args.ckpt_path, config)
    train_loader, val_loader = get_dataloaders(config)
    
    # Print the shape of the first spectrogram in the training set
    spectrogram, _ = next(iter(train_loader))
    print("Shape of the first spectrogram in the training set:", spectrogram.shape)

    training_pipeline(config, logger, model, train_loader, val_loader)


def get_model(ckpt, config):
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if ckpt:
        print('Loading from checkpoint')
        model = LightningViT.load_from_checkpoint(ckpt, config=config)
    else:
        model = LightningViT(config)
    model.to(device)
    return model


def get_dataloaders(config):
    # Make datasets

    train_set = SpectrogramDataset(manifest_path=config['tr_manifest_path'], 
                                   labels_map=config['labels_map'], 
                                   audio_config=config['audio_config'], 
                                   preload_data=config['preload_data'])
    val_set = SpectrogramDataset(manifest_path=config['val_manifest_path'], 
                                 labels_map=config['labels_map'], 
                                 audio_config=config['audio_config'],
                                 preload_data=config['preload_data'])

    # development mode (less files)
    if config['dev_mode']:
        print("Running dev_mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        train_set.files = train_set.files[:2000]
        train_set.len = len(train_set.files)
        #val_set.files = val_set.files[:50]
        #val_set.len = len(val_set.files)
        #config['hparams']['batch_size'] = 25

    # Make dataloaders - added shuffle to train_loader
    train_loader = DataLoader(train_set, batch_size=config['hparams']['batch_size'], shuffle=True, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=config['hparams']['batch_size'], num_workers=5)
    
    return train_loader, val_loader


def training_pipeline(config, logger, model, train_loader, val_loader):
    
    # Create callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", verbose=True)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=config['hparams']['early_stopping_patience'], verbose=True)
    callbacks = [model_checkpoint, early_stopping]

    trainer = L.Trainer(devices=4, accelerator="gpu", max_epochs=config['hparams']['n_epochs'], 
                        logger=logger,
                        callbacks=callbacks,
                        log_every_n_steps=100,
                        strategy='ddp_find_unused_parameters_true',
                        default_root_dir=config['exp']['save_dir'])

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    from argparse import ArgumentParser

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
            
    ap = ArgumentParser("Driver code")
    ap.add_argument('--config', type=str, required=True, help='Path to configuration file')
    ap.add_argument('--ckpt_path', type=str, help='Path to model checkpoint.')
    ap.add_argument('--id', type=str, help='Unique experiment identifier')
    ap.add_argument('--dev_mode', action='store_true', help='Flag to limit the dataset for testing purposes.')
    ap.add_argument('--preload_data', action='store_true', help='Flag to load dataset in memory.')
    args = ap.parse_args()

    main(args)