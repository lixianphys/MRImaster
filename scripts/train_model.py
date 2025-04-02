import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
from src.training.train import TrainCNN, TrainUnet, LiTrainCNN, LiTrainUnet
from src.utils.configYaml import (load_config_from_yaml, generate_param_combinations, pretty_print_config, extract_hyperparameters)
import click
from ignitetls.train import train_clf2d, train_seg3d

@click.command()
@click.option('--task', type=click.Choice(['clf2d', 'seg3d']), required=True, help="Task to train ('clf2d' or 'seg3d')")
@click.option('--config', type = click.Path(exists=True), required=True, help = "Path to the configuration YAML file")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="If set, only prints the generated configurations without running training."
)
@click.option('--framework', type=click.Choice(['ignite', 'torch', 'lightning']), default='ignite', help="Training framework to use.")
def main(task:str, config:str, dry_run:bool,framework:str):
    """
    CLI command to train a model.
    """
    # Load configuration
    if config.endswith('.yaml') or config.endswith('.yml'):
        config = load_config_from_yaml(config)
    elif config.endswith('.json'):
        import json
        with open(config, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Config file must be either YAML (.yaml/.yml) or JSON (.json)")



    # Dry run mode
    if dry_run:
        print("Dry run completed. No training was executed.")
        return
    # Run the experiments
    if framework == 'torch':
        click.echo("Switching to legacy model...")
        train_cnn = TrainCNN()
        train_unet = TrainUnet()
    elif framework == 'lightning':
        click.echo("Switching to Pytorch-lightning model...")
        train_cnn = LiTrainCNN()
        train_unet = LiTrainUnet()
    else:
        click.echo("Switching to Ignite model...")
        train_cnn = train_clf2d
        train_unet = train_seg3d
    
    if framework == 'ignite':

        if task.lower() == 'clf2d':
            click.echo("Training CNN model...")
            train_cnn(config)
        elif task.lower() == 'seg3d':
            click.echo("Training UNet model...")
            train_unet(config)
        else:
            click.echo("Invalid task type. Please specify 'clf2d' or 'seg3d'.")   

    else:
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

    for exp in experiments:
        # Determine which model to train
        if task.lower() == 'clf2d':
            click.echo("Training CNN model...")
            train_cnn(exp)
        elif task.lower() == 'seg3d':
            click.echo("Training UNet model...")
            train_unet(exp)
        else:
            click.echo("Invalid task type. Please specify 'clf2d' or 'seg3d'.")      


if __name__ == "__main__":
    main()











