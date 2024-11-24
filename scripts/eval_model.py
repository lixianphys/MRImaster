import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
from src.eval.evaluation import eval_cnn, eval_unet
from src.utils.utils import load_config_from_yaml
import click

@click.command()
@click.option('--model', type=click.Choice(['cnn', 'unet3d']), required=True, help="Model type to train ('cnn' or 'unet3d')")
@click.option('--config', type = click.Path(exists=True), required=True, help = "Path to the configuration YAML file")
@click.option('--data_path', type = click.Path(exists=True), default = '/',help="Path to the evaluation data directory.")
def eval_model(model,config,data_path):
    """
    CLI command to evaluate a model (CNN or UNet).
    """
    # Load configuration
    config = load_config_from_yaml(config)

    # Update evaluation data paths in the configuration if provided
    config['evaluation']['data_path'] = data_path

    # Determine which model to train
    if model.lower() == 'cnn':
        click.echo("Evaluating CNN model...")
        eval_cnn(config)
    elif model.lower() == 'unet3d':
        click.echo("Evaluating UNet model...")
        eval_unet(config)
    else:
        click.echo("Invalid model type. Please specify 'cnn' or 'unet3d'.")      

if __name__ == "__main__":
    eval_model()