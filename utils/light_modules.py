from typing import Any
import lightning as L
from torch import nn, optim

from transformers import get_cosine_schedule_with_warmup

class LightningKWT(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")
        self.model = model
        self.config = config
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        correct = outputs.argmax(1).eq(targets).sum()

        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"],
                            "tr_correct_predictions": correct}, on_epoch=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):    
        specs, targets = batch
        outputs = self(specs)
        val_loss = self.criterion(outputs, targets)
        correct = outputs.argmax(1).eq(targets).sum()
        accuracy = correct / len(batch)
        self.log_dict({"val_loss": val_loss, "val_correct_predictions": correct, "val_accuracy": accuracy}, on_epoch=True, on_step=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config["hparams"]["scheduler"]["n_warmup"], self.config["hparams"]["n_epochs"])
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]