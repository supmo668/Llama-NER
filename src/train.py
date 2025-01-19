import yaml
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, default_collate
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.accelerators import find_usable_cuda_devices

from src.lightning_module import NERLightningModule

def custom_collate(batch, device="cuda"):
    # Convert each field in the batch to a tensor and move to the specified device
    for i in range(len(batch)):
        for key in batch[i]:
            if isinstance(batch[i][key], list):
                batch[i][key] = torch.tensor(batch[i][key])
            else:
                batch[i][key] = batch[i][key]
    return default_collate(batch)

def custom_collate_fn(batch):
    return custom_collate(batch)

def run_training(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load datasets
    train_data = load_from_disk(os.path.join("data", "processed", "train"))
    val_data = load_from_disk(os.path.join("data", "processed", "val"))
    
    batch_size = cfg['data']['batch_size']
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    if cfg['training'].get('use_checkpoint', False):
        # Find latest checkpoint
        checkpoint_dir = cfg['training']['checkpoint_dir']
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Resuming from checkpoint: {checkpoint_path}")
            model = NERLightningModule.load_model_from_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found, starting training from scratch.")
            model = NERLightningModule(config_path)
    else:
        model = NERLightningModule(config_path)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg['training']['checkpoint_dir'],
        filename='ner-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg['training']['epochs'],
        accelerator="gpu" if cfg['training']['gpus'] > 0 else "cpu",
        devices=find_usable_cuda_devices(),
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=50
    )
    
    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    print(f"Training complete! Best model saved at: {checkpoint_callback.best_model_path}")
