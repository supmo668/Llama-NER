import yaml
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import pytorch_lightning as pl
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

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

def run_evaluation(config_path: str, split: str = 'test'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load dataset
    eval_data = load_from_disk(os.path.join("data", "processed", split))
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=cfg['data']['batch_size'],
        num_workers=4
    )
    
    # Load the model
    model = NERLightningModule(config_path)
    
    # If there's a checkpoint, load it
    checkpoint_dir = 'checkpoints'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            latest_checkpoint = max([os.path.join(checkpoint_dir, ckpt) for ckpt in checkpoints], key=os.path.getctime)
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
    
    # Convert to label strings (assuming you have a label list)
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]  # Example
    pred_labels = convert_predictions_to_labels(predictions, label_list)
    true_labels = convert_predictions_to_labels(references, label_list)
    
    # Calculate metrics
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    print(f"\nEvaluation on {split} set:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels))
