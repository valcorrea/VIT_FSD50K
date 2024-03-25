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
        mask = self.create_mask(spec)
        predictions, targets = self(spec, spec, mask)

        scale = math.sqrt(predictions.size(dim=-1))
        loss = self.criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)

        # self.model.ema_step()

        with torch.no_grad():
            target_var = self.compute_var(targets.float())
            prediction_var = self.compute_var(predictions.float())

        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"],
                            "target_var": target_var, "prediction_var": prediction_var}, on_epoch=True, on_step=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        self.model.ema_step()
    
    def validation_step(self, batch, batch_idx):
        spec = batch
        mask = self.create_mask(spec)
        predictions, targets = self(spec, spec, mask)
        
        scale = math.sqrt(predictions.size(dim=-1))
        loss = self.criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)

        target_var = self.compute_var(targets.float())
        prediction_var = self.compute_var(predictions.float())

        self.log_dict({"val_loss": loss,
                "val_target_var": target_var, "val_prediction_var": prediction_var}, on_epoch=True, on_step=True, sync_dist=True)        

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        return self.optimizer