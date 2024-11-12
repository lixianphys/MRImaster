import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
# predict.py
import click
import torch
import nibabel as nib
import numpy as np
from src.inference.predict import (
    load_cnn_model, load_unet3d_model,
    preprocess_image, preprocess_volume,
    cnn_inference, unet3d_inference,visualize_unet3d_prediction
)
import yaml
from src.utils.utils import CLA_label


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@click.command()
@click.option('--model_type', type=click.Choice(['cnn', 'unet3d']), required=True, help="Type of model to use for inference.")
@click.option('--config_path', type=click.Path(exists=True), required=True, help="Path to the configuration YAML file.")
def main(model_type, config_path):
    """
    CLI for running inference with a CNN or 3D UNet model.
    """
    # Load the configuration file
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if model_type == "cnn":
        cnn_config = config["cnn"]
        # Load model and preprocess input image
        model = load_cnn_model(cnn_config["model_path"], device, {
            "shape_in":tuple(cnn_config["shape_in"]),
            "num_classes":cnn_config["num_classes"],
            "initial_filters":cnn_config["initial_filters"],
            "num_fc1":cnn_config["num_fc1"],
            "dropout_rate":cnn_config["dropout_rate"]
        })
        image_tensor = preprocess_image(cnn_config["input_image_path"], device)
        
        # Run inference and print result
        prediction = cnn_inference(model, image_tensor)
        print(f"CNN Prediction: Class {CLA_label[prediction]}")

    elif model_type == "unet3d":
        unet3d_config = config["unet3d"]

        # Load model and preprocess input volume
        model = load_unet3d_model(unet3d_config["model_path"], device, unet3d_config["in_channels"], unet3d_config["out_channels"])
        volume_tensor = preprocess_volume(unet3d_config["input_volume_path"], device)
        
        # Run inference and save result
        prediction_volume = unet3d_inference(model, volume_tensor).astype(np.float32) # Nift1Image only accepts int16 or float32
        output_path = unet3d_config.get("output_path", "output/segmented_volume.nii")
        
        predicted_volume_nifti = nib.Nifti1Image(prediction_volume, affine=np.eye(4),dtype=np.int64)
        nib.save(predicted_volume_nifti, output_path)
        print(f"UNet3D Prediction saved to {output_path}")

        visualize_unet3d_prediction(volume_tensor,prediction_volume)

if __name__ == "__main__":
    main()
