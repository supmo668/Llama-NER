import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torchmetrics
from torchmetrics import classification as class_metrics
from torchmetrics.classification import MulticlassAccuracy
from transformers import get_cosine_schedule_with_warmup

from src.model import TokenClassificationModel
import yaml
from typing import Optional

class NERLightningModule(pl.LightningModule):
    """
    A PyTorch Lightning module for Named Entity Recognition (NER) using a token classification model.
    """
    def __init__(self, config_path: str = None):
        """Initialize the NERLightningModule."""
        super().__init__()
        self.automatic_optimization = True  # Enable automatic optimization
        
        if config_path:
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
            
            self.model = TokenClassificationModel(config_path)
            self.learning_rate = float(self.cfg['training']['lr'])
            self.save_hyperparameters()

        # Initialize metrics on the correct device
        self.test_metrics = None
        self.val_metrics = None
        
    def setup(self, stage: str):
        """Initialize metrics on the correct device."""
        if stage == 'fit' or stage == 'validate':
            self.val_metrics = torchmetrics.MetricCollection({
                "accuracy": MulticlassAccuracy(
                    num_classes=self.model.num_labels, 
                    ignore_index=-100
                ).to(self.device)
            }, prefix="val_")
            
        if stage == 'test':
            self.test_metrics = torchmetrics.MetricCollection({
                "precision": class_metrics.Precision(
                    task="multiclass", 
                    num_classes=self.model.num_labels, 
                    average='macro'
                ).to(self.device),
                "recall": class_metrics.Recall(
                    task="multiclass", 
                    num_classes=self.model.num_labels, 
                    average='macro'
                ).to(self.device),
                "f1": class_metrics.F1Score(
                    task="multiclass", 
                    num_classes=self.model.num_labels, 
                    average='macro'
                ).to(self.device),
                "accuracy": MulticlassAccuracy(
                    num_classes=self.model.num_labels, 
                    ignore_index=-100
                ).to(self.device)
            }, prefix="test_")

    def load_model_from_checkpoint(checkpoint_path: str) -> 'NERLightningModule':
        """
        Load a model from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            NERLightningModule: The loaded model.
        """
        return NERLightningModule.load_from_checkpoint(checkpoint_path)

    def forward(self, batch: dict) -> dict:
        """
        Forward pass through the model.

        Args:
            batch (dict): A batch of input data.

        Returns:
            dict: The model's output.
        """
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels", None)
        )
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (dict): A batch of input data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The training loss.
        """
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.
        """
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("val_loss", loss, prog_bar=True)
        
        # Get predictions based on whether CRF is used
        if hasattr(self.model, 'use_crf') and self.model.use_crf:
            # Use CRF decoded tags
            decoded_tags = outputs["decoded_tags"]
            batch_size, seq_length = batch["labels"].shape
            preds = torch.full((batch_size, seq_length), fill_value=0, 
                             dtype=torch.long, device=batch["labels"].device)
            
            # Convert decoded_tags to tensor
            for i, seq in enumerate(decoded_tags):
                seq_len = min(len(seq), seq_length)
                preds[i, :seq_len] = torch.tensor(seq[:seq_len], 
                                                dtype=torch.long, 
                                                device=batch["labels"].device)
        else:
            # Standard argmax prediction
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
        
        # Create mask for valid positions
        mask = batch["labels"] != -100
        
        # Get valid predictions and labels
        valid_preds = preds[mask]
        valid_labels = batch["labels"][mask]
        
        # Update metrics
        if len(valid_preds) > 0 and len(valid_labels) > 0:
            self.val_metrics.update(valid_preds, valid_labels)
        
        return loss

    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        if self.val_metrics is not None:
            try:
                metrics = self.val_metrics.compute()
                self.val_metrics.reset()
                
                # Log all metrics
                for name, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.log(name, value, prog_bar=True)
            except Exception as e:
                print(f"Error in validation metrics computation: {e}")
                self.val_metrics.reset()

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform a single test step.

        Args:
            batch (dict): A batch of input data containing:
                - input_ids: Tensor of shape (batch_size, seq_length)
                - attention_mask: Tensor of shape (batch_size, seq_length)
                - labels: Tensor of shape (batch_size, seq_length)
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The test loss (scalar).
        """
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("test_loss", loss, prog_bar=True)

        # Create mask for valid positions (exclude padding and special tokens)
        mask = batch["labels"] != -100  # Shape: (batch_size, seq_length)
        
        if hasattr(self.model, 'use_crf') and self.model.use_crf:
            # Use CRF decoding
            # decoded_tags List[List[int]] will be a list of lists where each inner list has length seq_length
            decoded_tags = outputs["decoded_tags"]
            
            # Convert decoded_tags to tensor matching the original shape
            batch_size, seq_length = batch["labels"].shape
            preds = torch.full((batch_size, seq_length), fill_value=0, 
                             dtype=torch.long, device=batch["labels"].device)
            
            # Convert decoded_tags to tensor
            for i, seq in enumerate(decoded_tags):
                seq_len = min(len(seq), seq_length)
                preds[i, :seq_len] = torch.tensor(seq[:seq_len], 
                                                dtype=torch.long, 
                                                device=batch["labels"].device)
        else:
            # Standard argmax prediction
            logits = outputs["logits"]  # Shape: (batch_size, seq_length, num_labels)
            preds = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_length)
        
        # Get valid predictions and labels
        valid_preds = preds[mask]  # Only take predictions where mask is True
        valid_labels = batch["labels"][mask]  # Only take labels where mask is True
        
        # Update metrics
        if len(valid_preds) > 0 and len(valid_labels) > 0:
            self.test_metrics.update(valid_preds, valid_labels)

        return loss

    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        if self.test_metrics is not None:
            try:
                metrics = self.test_metrics.compute()
                self.test_metrics.reset()
                
                # Log all metrics
                for name, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.log(name, value, prog_bar=True)
            except Exception as e:
                print(f"Error in test metrics computation: {e}")
                self.test_metrics.reset()

    def configure_optimizers(self) -> dict:
        """
        Configure the optimizers and learning rate schedulers.
        
        Using recommended hyperparameters for NER fine-tuning:
        - Learning rate: 5e-5 (from config)
        - Weight decay: 0.001 (less aggressive regularization)
        - Beta1: 0.9 (standard momentum)
        - Beta2: 0.999 (standard second moment)
        - Epsilon: 1e-8 (numerical stability)
        
        Returns:
            dict: A dictionary containing the optimizer and scheduler.
        """
        # Different weight decay for different parameter groups
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.001
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Gradual warmup and linear decay
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = num_training_steps // 10  # 10% of training for warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def teardown(self, stage: Optional[str] = None):
        """Clean up resources when the module is destroyed."""
        super().teardown(stage)
        
        # Clean up metrics
        if self.test_metrics is not None:
            self.test_metrics.reset()
            for metric in self.test_metrics.values():
                metric.cpu()
                del metric
            self.test_metrics = None
            
        if self.val_metrics is not None:
            self.val_metrics.reset()
            for metric in self.val_metrics.values():
                metric.cpu()
                del metric
            self.val_metrics = None
            
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force garbage collection
        import gc
        gc.collect()
