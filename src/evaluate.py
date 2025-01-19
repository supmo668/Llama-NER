import yaml
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
import pytorch_lightning as pl
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from pathlib import Path

from src.lightning_module import NERLightningModule

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
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load dataset
    eval_data = load_from_disk(Path("data/processed") / split)
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=cfg['data']['batch_size'],
        num_workers=4
    )
    
    # Load the model
    model = NERLightningModule(config_path)
    
    # If there's a checkpoint, load it
    checkpoint_dir = Path('checkpoints')
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.ckpt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda ckpt: ckpt.stat().st_ctime)
            print(f"Loading checkpoint: {latest_checkpoint}")
            model = model.load_from_checkpoint(latest_checkpoint, config_path=config_path)
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu" if cfg['training']['gpus'] > 0 else "cpu",
        devices=cfg['training']['gpus'] if cfg['training']['gpus'] > 0 else None,
    )
    
    # Get predictions
    predictions = []
    references = []
    model.eval()
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(batch)
            
            # Get predictions
            if model.model.crf is not None:
                preds = outputs["logits"]
            else:
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
            
            # Move to CPU and convert to numpy
            preds = preds.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            
            # Filter out padding (-100)
            for pred_seq, label_seq in zip(preds, labels):
                valid_pred = []
                valid_label = []
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:  # Ignore padding
                        valid_pred.append(p)
                        valid_label.append(l)
                predictions.append(valid_pred)
                references.append(valid_label)
    
    # Load label list from the saved file
    label_list_path = Path('data/processed/label_list.txt')
    with label_list_path.open('r') as label_file:
        label_list = [line.strip() for line in label_file]
    
    metrics = calculate_metrics(predictions, references, label_list)
    
    print(f"\nEvaluation on {split} set:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(metrics['classification_report'])
