from typing import Any
import math
import torch
from torch import nn, optim
import lightning as L
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import AveragePrecision
from timm.models import VisionTransformer

from src.models.KWT import KWT

class LightningViT(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Get Hyperparameters
        self.n_epochs = config['hparams']['n_epochs']
        self.img_size = config['hparams']['KWT']['input_res']
        self.patch_size = config['hparams']['KWT']['patch_res']
        self.num_classes = config['hparams']['KWT']['num_classes']
        self.embed_dim = config['hparams']['KWT']['dim']
        self.mlp_ratio = config['hparams']['KWT']['mlp_dim'] // self.embed_dim
        self.num_heads = config['hparams']['KWT']['heads']
        self.depth = config['hparams']['KWT']['depth']
        self.cw = config.get('cw', None)
        self.lr = config['hparams']['optimizer']['lr']
        self.betas = config['hparams']['optimizer']['betas']
        self.eps = config['hparams']['optimizer']['eps']
        self.weight_decay = config['hparams']['optimizer']['weight_decay']
        self.n_warmup = config['hparams']['scheduler']['n_warmup']
        
        # Initialize model
        self.model = self.get_model()

        # Loss function
        self.criterion = self.get_criterion()

        # Metrics
        self.train_precision = AveragePrecision(task="multilabel", num_labels=200)
        self.val_precision = AveragePrecision(task="multilabel", num_labels=200)

    def get_criterion(self):
        if self.cw is not None:
            print("!!!!!!!!!!! loading class weights !!!!!!!!")
            self.cw = torch.load(self.cw, map_location="cpu")
            print("self.cw shape:", self.cw.shape)
        criterion =  nn.BCEWithLogitsLoss(pos_weight=self.cw)
        return criterion
    
    def get_model(self):
        model = VisionTransformer(img_size=self.img_size,
                                  patch_size = self.patch_size,
                                  in_chans = 1,
                                  num_classes = self.num_classes,
                                  embed_dim = self.embed_dim,
                                  depth = self.depth,
                                  num_heads = self.num_heads,
                                  mlp_ratio = self.mlp_ratio
                                       )
        return model

    def forward(self, specs):
        return self.model(specs)

    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        y_pred_sigmoid = torch.sigmoid(outputs) #predictions
        self.train_precision(y_pred_sigmoid,targets.long())
        self.log("train_mAP", self.train_precision, prog_bar=True, on_epoch=True, synch_dist=True)
        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_epoch=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):    
        specs, targets = batch
        outputs = self(specs)
        val_loss = self.criterion(outputs, targets)
        y_pred_sigmoid = torch.sigmoid(outputs)
        self.val_precision(y_pred_sigmoid,targets.long())
        self.log("val_mAP",self.val_precision, prog_bar=True, on_epoch=True, synch_dist=True)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, sync_dist=True)
        return val_loss
    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                           betas=self.betas,
                           eps=self.eps,
                           weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.n_warmup, self.n_epochs)
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class LightningKWT(L.LightningModule):
    def __init__(self,
                 config):
        super().__init__()
        
        self.model = KWT(**config['hparams']['KWT']) 
        self.config = config
        self.train_precision = AveragePrecision(task="multilabel", num_labels=200) #logging average precision
        self.val_precision = AveragePrecision(task="multilabel", num_labels=200) #logging average precision
        if self.config["cw"] is not None:
            print("!!!!!!!!!!! loading class weights !!!!!!!!")
            self.cw = torch.load(self.config["cw"], map_location="cpu")
            print("self.cw shape:", self.cw.shape)
        else:
            self.cw = None
        self.criterion =  nn.BCEWithLogitsLoss(pos_weight=self.cw) # multi label classification
        
    def forward(self, specs):
        return self.model(specs)
    
    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        #correct = outputs.argmax(1).eq(targets.argmax(1)).sum().float()
        y_pred_sigmoid = torch.sigmoid(outputs) #predictions
        self.train_precision(y_pred_sigmoid,targets.long())
        #auc = torch.tensor(AveragePrecision(targets.detach().cpu().numpy(),
        #                                           y_pred_sigmoid.detach().cpu().numpy(), average="macro"))
        self.log("train_mAP", self.train_precision, prog_bar=True, on_epoch=True, synch_dist=True)

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
        self.log("val_mAP",self.val_precision, prog_bar=True, on_epoch=True, synch_dist=True)

        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, sync_dist=True)
    

        return val_loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        #scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config["hparams"]["scheduler"]["n_warmup"], self.config["hparams"]["n_epochs"])
        return [self.optimizer]#, [{"scheduler": scheduler, "interval": "epoch"}]
 
    
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