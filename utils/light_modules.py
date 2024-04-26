from typing import Any
import math
import torch
from torch import nn, optim
import lightning as L
from transformers import get_cosine_schedule_with_warmup

from src.models.KWT import KWT

class LightningViT(L.LightningModule):
    

class LightningSweep(L.LightningModule):
    def __init__(self,
                 hparams,
                 arch):
        super().__init__()
        self.criterion =  nn.BCEWithLogitsLoss() # multi label classification
        self.model = KWT(**arch)
        self.optimizer = hparams['optimizer']
        self.scheduler = hparams['scheduler']
        self.learning_rate = hparams['learning_rate']
        self.n_warmup = hparams['n_warmup']
        self.n_epochs = hparams['n_epochs']

        
    def forward(self, specs):
        return self.model(specs)
    
    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_epoch=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):    
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        self.log_dict({"val_loss": loss}, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        if self.scheduler:
            scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.n_warmup, self.n_epochs)
            return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return self.optimizer