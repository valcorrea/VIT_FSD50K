import math
from typing import Any

import lightning as L
import torch
from src.models.KWT import KWT, KWTFNet
from torch import nn, optim
from transformers import get_cosine_schedule_with_warmup


class LightningKWT(L.LightningModule):
    def __init__(self, config, useFnet=False):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()  # multi label classification
        self.model = (
            KWT(**config["hparams"]["KWT"])
            if not useFnet
            else KWTFNet(**config["hparams"]["KWTFNet"])
        )
        self.config = config

    def forward(self, specs):
        return self.model(specs)

    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        self.log_dict(
            {
                "train_loss": loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "tr_correct_predictions": correct,
            },
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        val_loss = self.criterion(outputs, targets)
        correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        accuracy = correct / targets.shape[0]
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_correct_predictions": correct,
                "val_accuracy": accuracy,
            },
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        return val_loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["hparams"]["optimizer"]["lr"],
            betas=self.config["hparams"]["optimizer"]["betas"],
            eps=self.config["hparams"]["optimizer"]["eps"],
            weight_decay=self.config["hparams"]["optimizer"]["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            self.config["hparams"]["scheduler"]["n_warmup"],
            self.config["hparams"]["n_epochs"],
        )
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
