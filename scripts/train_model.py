""" 
Build tools for training models.
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
import pandas as pd
from src.training.train import train_cnn, train_unet
import click
import yaml


def load_config(config_path):
    """Load configuration from a YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@click.command()
@click.option('--model', type=click.Choice(['cnn', 'unet3d']), required=True, help="Model type to train ('cnn' or 'unet')")
@click.option('--config', type = click.Path(exists=True), required=True, help = "Path to the configuration YAML file")
@click.option('--data_path', type = click.Path(exists=True), default = '/',help="Path to the training data directory.")
@click.option('--use_mlflow', is_flag=True, help="Use MLflow for tracking training metrics and parameters.")
def train_model(model,config,data_path,use_mlflow):
    """
    CLI command to train a model (CNN or UNet).
    """
    # Load configuration
    config = load_config(config)

    # Update data paths in the configuration if provided
    config['data']['train_path'] = data_path + '/train'
    config['data']['val_path'] = data_path + '/val'

    if use_mlflow:
        config['mlflow']['enable'] = True
    # Determine which model to train
    if model.lower() == 'cnn':
        click.echo("Training CNN model...")
        train_cnn(config)
    elif model.lower() == 'unet3d':
        click.echo("Training UNet model...")
        train_unet(config)
    else:
        click.echo("Invalid model type. Please specify 'cnn' or 'unet'.")      


# kaggle_link = "sartajbhuvaji/brain-tumor-classification-mri"
# dir_to_store = "data/raw_data/brain-tumor-classification-mri/"

# brain_tumor_dt = KaggleDataPipe(kaggle_link,dir_to_store)
# brain_tumor_dt.load_from_kaggle()
# labels = brain_tumor_dt.get_labels("Training")

# DATASET = os.path.join(dir_to_store,"Training")
# OUTPUT = "data/processed_data/brain-tumor-classification-mri"


if __name__ == "__main__":
    train_model()











