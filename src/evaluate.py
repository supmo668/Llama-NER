import yaml
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_from_disk
from pathlib import Path
from src.lightning_module import NERLightningModule
from src.dataloader import get_dataloader
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch


def load_evaluation_data(config_path: str, split: str = 'test'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    eval_data = load_from_disk(Path("data/processed") / split)
    eval_dataloader = get_dataloader(eval_data, cfg['data']['batch_size'])
    return eval_dataloader, cfg


def initialize_model(cfg, config_path):
    if cfg['evaluation'].get('load_checkpoint', True):
        checkpoint_dir = Path(cfg['training']['checkpoint_dir'])
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.ckpt'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda ckpt: ckpt.stat().st_ctime)
                print(f"Loading checkpoint for evaluation: {latest_checkpoint}")
                model = NERLightningModule.load_from_checkpoint(latest_checkpoint, config_path=config_path)
            else:
                print("No checkpoint found, starting evaluation from scratch.")
                model = NERLightningModule(config_path)
        else:
            print("Checkpoint directory does not exist.")
            model = NERLightningModule(config_path)
    else:
        model = NERLightningModule(config_path)
    return model


def convert_predictions_to_labels(predictions, label_list):
    """Convert numeric predictions to label strings."""
    converted = []
    for pred_seq in predictions:
        converted_seq = []
        for p in pred_seq:
            if p == -100:
                continue
            converted_seq.append(label_list[p])
        converted.append(converted_seq)
    return converted


def calculate_metrics(predictions, references, label_list):
    """
    Calculate evaluation metrics for NER predictions.
    
    Args:
        predictions: List of predicted label sequences.
        references: List of true label sequences.
        label_list: List of all possible labels.
    
    Returns:
        A dictionary containing precision, recall, and F1 score.
    """
    pred_labels = convert_predictions_to_labels(predictions, label_list)
    true_labels = convert_predictions_to_labels(references, label_list)

    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": classification_report(true_labels, pred_labels)
    }


def run_evaluation(config_path: str, split: str = 'test'):
    eval_dataloader, cfg = load_evaluation_data(config_path, split)
    model = initialize_model(cfg, config_path)

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu" if cfg['training']['gpus'] > 0 else "cpu",
        devices=1,  # Recommended to use a single device for testing
        log_every_n_steps=50
    )

    # Test the model
    trainer.test(model, dataloaders=eval_dataloader)

    print("Evaluation complete!")