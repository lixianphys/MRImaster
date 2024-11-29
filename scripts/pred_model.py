import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
# predict.py
import click
from src.inference.predict import pred_cnn, pred_unet
from src.utils.configYaml import load_config_from_yaml


@click.command()
@click.option('--model', type=click.Choice(['cnn', 'unet3d']), required=True, help="Model type to predict ('cnn' or 'unet')")
@click.option('--config', type = click.Path(exists=True), required=True, help = "Path to the configuration YAML file")
def pred_model(model, config):
    """
    CLI for running inference with a CNN or 3D UNet model.
    """
    # Load the configuration file
    config = load_config_from_yaml(config)
    
    if model.lower() == 'cnn':
        # Load model and preprocess input image
        pred_cnn(config)

    elif model.lower() == 'unet3d':
        pred_unet(config)


if __name__ == "__main__":
    pred_model()
