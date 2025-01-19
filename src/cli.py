import click
from src import data_prep, train, evaluate
import yaml
import os
from src.lightning_module import NERLightningModule

from dotenv import load_dotenv
load_dotenv()

@click.group()
def cli():
    """CLI entry point for the NER pipeline."""
    pass

@cli.command()
@click.option('--config-path', default='config/config.yaml', help='Path to the config file.')
@click.option('--ratio', default=1.0, help='Fraction of the dataset to retain (default is 1.0, meaning no reduction).')
def prepare_data(config_path, ratio):
    """Prepare and process the data for NER training."""
    data_prep.prepare_data(config_path, ratio)

@cli.command()
@click.option('--config-path', default='config/config.yaml', help='Path to the config file.')
def run_train(config_path):
    """Train (fine-tune) the NER model."""
    train.run_training(config_path)

@cli.command()
@click.option('--config-path', default='config/config.yaml', help='Path to the config file.')
@click.option('--split', default='test', help='Dataset split to evaluate on.')
def run_evaluate(config_path, split):
    """Evaluate the NER model on a specified split."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg['evaluation'].get('load_checkpoint', True):
        checkpoint_dir = cfg['training']['checkpoint_dir']
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            latest_checkpoint = max([os.path.join(checkpoint_dir, ckpt) for ckpt in checkpoints], key=os.path.getctime)
            print(f"Loading checkpoint for evaluation: {latest_checkpoint}")
            model = NERLightningModule.load_model_from_checkpoint(latest_checkpoint)
        else:
            print("No checkpoint found for evaluation, using initial model.")
            model = NERLightningModule(config_path)
    else:
        model = NERLightningModule(config_path)

    evaluate.run_evaluation(config_path, split)

if __name__ == '__main__':
    cli()
