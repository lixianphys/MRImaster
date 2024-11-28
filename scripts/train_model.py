import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
from src.training.train import TrainCNN, TrainUnet
from src.utils.configYaml import (load_config_from_yaml, generate_param_combinations, pretty_print_config, extract_hyperparameters)
import click

@click.command()
@click.option('--model', type=click.Choice(['cnn', 'unet3d']), required=True, help="Model type to train ('cnn' or 'unet')")
@click.option('--config', type = click.Path(exists=True), required=True, help = "Path to the configuration YAML file")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="If set, only prints the generated configurations without running training."
)
@click.option('--use_mlflow', is_flag=True, help="Use MLflow for tracking training metrics and parameters.")
def train_model(model:str, config:str, dry_run:bool, use_mlflow:bool):
    """
    CLI command to train a model.
    """
    # Load configuration
    config = load_config_from_yaml(config)

    # Generate parameter combinations
    experiments = generate_param_combinations(config,ignore_keys=['eval','deploy'])
    print(f"Run {len(experiments)} experiments:")

    # Print hyperparameters for each configuration
    for num, exp in enumerate(experiments):
        print(f" # {num+1} experiment:")
        extracted_params = extract_hyperparameters(exp,[
        'model',
        'train.load.train_ratio',
        'train.batch_size',
        'train.epochs',
        'train.learning_rate',
        ]
        )
        pretty_print_config(extracted_params)
        print("---------------------")

    # Dry run mode
    if dry_run:
        print("Dry run completed. No training was executed.")
        return
    # Run the experiments
    train_cnn = TrainCNN()
    train_unet = TrainUnet()
    for exp in experiments:
        if use_mlflow:
            exp['train']['mlflow']['enabled'] = True
        # Determine which model to train
        if model.lower() == 'cnn':
            click.echo("Training CNN model...")
            train_cnn(exp)
        elif model.lower() == 'unet3d':
            click.echo("Training UNet model...")
            train_unet(exp)
        else:
            click.echo("Invalid model type. Please specify 'cnn' or 'unet'.")      


if __name__ == "__main__":
    train_model()











