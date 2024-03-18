import lightning as L
import torch

from torch import nn, optim
from src.models.ssformer import SSTransformer
import math
from src.utils.masking import AudioMaskingGenerator

class LightningTransformer(L.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 config,
                 **kwargs):
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
            

    def training_step(self, batch, batch_idx):
        input = batch
        input = input.expand(1, -1, -1, -1) # Create an extra empty dimension
        input = input.permute(1, 0, 2, 3)
        batch_size = input.size(dim=0)
        audio_length = input.size(dim=-1)
        mask = self.mask_generator(shape=(batch_size, audio_length))
        mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()
        predictions, targets = self(input, input, mask)
        scale = math.sqrt(predictions.size(dim=-1))
        loss = self.criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)
        self.model.ema_step()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        return optimizer