import lightning as L
import torch

from torch import nn, optim
from src.models.ssformer import SSTransformer
import math
from src.utils.masking import AudioMaskingGenerator

class LightningTransformer(L.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 config):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")
        self.model = SSTransformer(encoder, **config['hparams']['SSformer'])
        self.config = config
        self.mask_generator = AudioMaskingGenerator(mask_prob=config["hparams"]["SSformer"]["mask_prob"],
                                           mask_length=config["hparams"]["SSformer"]["mask_length"],
                                           attention_mask=None,
                                           min_masks=config["hparams"]["SSformer"]["min_masks"])
        
    def forward(self, inputs, target, mask):
        return self.model(inputs, target, mask)
            
    def create_mask(self, spec):
        batch_size = spec.size(dim=0)
        audio_length = spec.size(dim=-1)
        mask = self.mask_generator(shape=(batch_size, audio_length))
        mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()
        return mask
    
    def transform(self, spec):
        spec = spec.expand(1, -1, -1, -1)
        spec = spec.permute(1, 0, 2, 3)
        return spec
    
    def compute_var(self, y: torch.Tensor):
        """
        Function for computing standard deviation of target
        :param y:
        :return: standard deviation of y
        """
        y = y.view(-1, y.size(-1))
        return torch.sqrt(y.var(dim=0) + 1e-6).mean()
    
    def training_step(self, batch, batch_idx):
        spec = batch
        spec = self.transform(spec)
        mask = self.create_mask(spec)
        predictions, targets = self(spec, spec, mask)

        scale = math.sqrt(predictions.size(dim=-1))
        loss = self.criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)

        self.model.ema_step()
        return loss
    
    def on_validation_epoch_start(self):
        self.running_loss = 0.0
        self.running_target_var = 0.0
        self.running_prediction_var = 0.0

    def validation_step(self, batch, batch_idx):
        spec = batch
        spec = self.transform(spec)
        mask = self.create_mask(spec)
        predictions, targets = self(spec, spec, mask)
        
        scale = math.sqrt(predictions.size(dim=-1))
        loss = self.criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)

        target_var = self.compute_var(targets.float())
        prediction_var = self.compute_var(predictions.float())

        self.running_loss += loss.item()
        self.running_target_var += target_var.item()
        self.running_prediction_var += prediction_var.item()
        
    def on_validation_epoch_end(self):
        avg_loss = self.running_loss / len(self.trainer.val_dataloaders.dataset)
        avg_target_var = self.running_target_var / len(self.trainer.val_dataloaders)
        avg_prediction_var = self.running_prediction_var / len(self.trainer.val_dataloaders)
        self.log_dict({"val_loss": avg_loss,
                "avg_val_target_var": avg_target_var, "avg_val_prediction_var": avg_prediction_var}, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        return optimizer