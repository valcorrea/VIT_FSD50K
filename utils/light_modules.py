from typing import Any
import math
import torch
from torch import nn, optim
import lightning as L
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import AveragePrecision

from src.models.KWT import KWT

class LightningKWT(L.LightningModule):
    def __init__(self,
                 config):
        super().__init__()
        self.criterion =  nn.BCEWithLogitsLoss() # multi label classification
        self.model = KWT(**config['hparams']['KWT']) 
        self.config = config
        self.train_precision = AveragePrecision(task="multilabel", num_labels=200) #logging average precision
        self.val_precision = AveragePrecision(task="multilabel", num_labels=200) #logging average precision
        
    def forward(self, specs):
        return self.model(specs)
    
    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        #correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        y_pred_sigmoid = torch.sigmoid(outputs)
        self.train_precision(y_pred_sigmoid,targets.long())
        #auc = torch.tensor(AveragePrecision(targets.detach().cpu().numpy(),
        #                                           y_pred_sigmoid.detach().cpu().numpy(), average="macro"))
        self.log("train_mAP", self.train_precision, prog_bar=True, on_epoch=True)

        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_epoch=True, on_step=True, sync_dist=True)
                            #"tr_correct_predictions": correct}, on_epoch=True, on_step=True, sync_dist=True)
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):    
        specs, targets = batch
        outputs = self(specs)
        val_loss = self.criterion(outputs, targets)
        y_pred_sigmoid = torch.sigmoid(outputs)
        self.val_precision(y_pred_sigmoid,targets.long())
        #correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        #accuracy = correct / targets.shape[0]
        self.log("val_mAP",self.val_precision, prog_bar=True, on_epoch=True)
        
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, sync_dist=True)
    

        return val_loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config["hparams"]["scheduler"]["n_warmup"], self.config["hparams"]["n_epochs"])
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]