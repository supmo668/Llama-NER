import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from src.model import TokenClassificationModel
import yaml

class NERLightningModule(pl.LightningModule):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.model = TokenClassificationModel(config_path)
        self.learning_rate = float(self.cfg['training']['lr'])
        self.save_hyperparameters()
    
    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels", None)
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
