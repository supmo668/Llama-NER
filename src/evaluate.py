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
import pandas as pd
from pytorch_lightning.callbacks import Callback
from datetime import datetime

def load_evaluation_data(config_path: str, split: str = 'test'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    eval_data = load_from_disk(Path("data/processed") / split)
    eval_dataloader = get_dataloader(eval_data, cfg['data']['batch_size'], num_workers=1)
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


class MetricsCallback(Callback):
    """Callback to save metrics to CSV file after testing."""
    
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.test_metrics = {}
        
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: 'LightningModule') -> None:
        """Collect metrics at the end of test epoch."""
        metrics = trainer.callback_metrics
        
        # Convert tensor values to python scalars and format metric names
        self.test_metrics = {}
        for k, v in metrics.items():
            # Remove 'test_' prefix for cleaner names
            metric_name = k.replace('test_', '') if k.startswith('test_') else k
            # Convert tensor to scalar
            metric_value = v.item() if hasattr(v, 'item') else v
            # Format float values
            if isinstance(metric_value, float):
                metric_value = f"{metric_value:.4f}"
            self.test_metrics[metric_name] = metric_value
        
    def save_metrics(self) -> None:
        """Save the collected metrics to a CSV file."""
        if not self.test_metrics:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f'test_metrics_{timestamp}.csv')
        
        # Add timestamp to metrics
        metrics_with_time = {
            'timestamp': timestamp,
            **self.test_metrics
        }
        
        # Convert metrics to DataFrame
        df = pd.DataFrame([metrics_with_time])
        
        # Reorder columns to put important metrics first
        important_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
        columns = ['timestamp'] + important_metrics + [
            col for col in df.columns 
            if col not in important_metrics + ['timestamp']
        ]
        df = df.reindex(columns=columns)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nMetrics saved to: {output_file}")
        print("\nTest Metrics Summary:")
        for metric in important_metrics:
            if metric in self.test_metrics:
                print(f"{metric.capitalize()}: {self.test_metrics[metric]}")


def run_evaluation(config_path: str, split: str = 'test'):
    """
    Run evaluation and ensure proper cleanup of resources.
    """
    
    eval_dataloader, cfg = load_evaluation_data(config_path, split)
    model = initialize_model(cfg, config_path)
    
    # Initialize trainer with metrics callback
    trainer = pl.Trainer(
        accelerator="gpu" if cfg['training']['gpus'] > 0 else "cpu",
        devices=1,  # Recommended to use a single device for testing
        log_every_n_steps=200
    )

    # Test the model
    trainer.test(model=model, datamodule=eval_dataloader)
    
        
if __name__ == '__main__':
    # Set multiprocessing start method
    run_evaluation('path_to_config.yaml')