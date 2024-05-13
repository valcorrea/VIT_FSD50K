import math
from typing import Any

import lightning as L
import torch
from src.models.KWT import KWT, KWTFNet
from torch import nn, optim
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import AveragePrecision, MulticlassAccuracy


class LightningKWT(L.LightningModule):
    def __init__(self, config, useFnet=False):
        super().__init__()
        self.model = (
            KWT(**config["hparams"]["KWT"])
            if not useFnet
            else KWTFNet(**config["hparams"]["KWTFNet"])
        )
        self.config = config
        self.num_classes= (
            self.config['hparams']['KWT']['num_classes']
            if not useFnet
            else self.config['hparams']['KWTFNet']['num_classes']
        )
        # if self.config.get('cw', None) is not None:
        #     print("!!!!!!!!!!! loading class weights !!!!!!!!")
        #     self.cw = torch.load(self.config["cw"], map_location="cpu")
        #     print("self.cw shape:", self.cw.shape)
        # else:
        #     self.cw = None
        # if self.config["mode"] == "multilabel":
        #     self.train_precision = AveragePrecision(task="multilabel", num_labels=self.num_classes) #logging average precision
        #     self.val_precision = AveragePrecision(task="multilabel", num_labels=self.num_classes) #logging average precision
        #     self.criterion =  nn.BCEWithLogitsLoss(pos_weight=self.cw) # multi label classification
        # else:
        self.train_precision = MulticlassAccuracy(num_classes=self.num_classes) #logging multiclass accuracy
        self.val_precision = MulticlassAccuracy(num_classes=self.num_classes) #logging multiclass accuracy
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, specs):
        return self.model(specs)

    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        #correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        # if self.config["mode"] == "multilabel":
        #     y_pred_sigmoid = torch.sigmoid(outputs) #predictions
        #     self.train_precision(y_pred_sigmoid,targets.long())
        #     self.log("train_mAP", self.train_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        # else:
        self.train_precision(outputs,targets)
        self.log("train_acc", self.train_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        #auc = torch.tensor(AveragePrecision(targets.detach().cpu().numpy(),
        #                                           y_pred_sigmoid.detach().cpu().numpy(), average="macro"))
    
        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_epoch=True, on_step=True, sync_dist=True)
                            #"tr_correct_predictions": correct}, on_epoch=True, on_step=True, sync_dist=True)
        
        
        return loss

    def validation_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        val_loss = self.criterion(outputs, targets)
        # if self.config["mode"] == "multilabel":
        #     y_pred_sigmoid = torch.sigmoid(outputs)
        #     self.val_precision(y_pred_sigmoid,targets.long())
        #     self.log("val_mAP",self.val_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        # else:
        self.val_precision(outputs, targets)
        self.log("val_acc",self.val_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        #correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        #accuracy = correct / targets.shape[0]
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, sync_dist=True)
        return val_loss
   
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        specs, targets = batch
        outputs = self(specs)
        test_loss = self.criterion(outputs, targets)
        # if self.config["mode"] == "multilabel":
        #     y_pred_sigmoid = torch.sigmoid(outputs)
        #     self.val_precision(y_pred_sigmoid,targets.long())
        #     self.log("test_mAP",self.val_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        # else:
        self.val_precision(outputs, targets)
        self.log("test_acc",self.val_precision, prog_bar=True, on_epoch=True, sync_dist=True)
        #correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        #accuracy = correct / targets.shape[0]
        self.log_dict({"test_loss": test_loss}, on_epoch=True, on_step=True, sync_dist=True)
        return test_loss
    
    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_test_loss, sync_dist=True)
        
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config["hparams"]["scheduler"]["n_warmup"], self.config["hparams"]["n_epochs"])
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
 
    
class LightningSweep(L.LightningModule):
    def __init__(self,
                 hparams,
                 arch):
        super().__init__()
        self.model = KWT(**arch)
        self.optimizer = hparams['optimizer']
        self.scheduler = hparams['scheduler']
        self.learning_rate = hparams['learning_rate']
        self.n_warmup = hparams['n_warmup']
        self.n_epochs = hparams['n_epochs']
        self.weight_decay = hparams['weight_decay']
        self.num_classes = 35

        self.train_precision = MulticlassAccuracy(num_classes=self.num_classes) #logging multiclass accuracy
        self.val_precision = MulticlassAccuracy(num_classes=self.num_classes) #logging multiclass accuracy
        self.criterion = nn.CrossEntropyLoss()

        
    def forward(self, specs):
        return self.model(specs)
    
    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        acc = self.train_precision(outputs,targets)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)    
        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_epoch=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):    
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        acc = self.train_precision(outputs, targets)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log_dict({"val_loss": loss}, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.learning_rate, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        if self.scheduler:
            scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.n_warmup, self.n_epochs)
            return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return self.optimizer
