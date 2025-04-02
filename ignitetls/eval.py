import logging
import sys
import os
import monai
import torch
import mlflow
import numpy as np
from pathlib import Path
from ignite.metrics import Accuracy

from monai.apps import get_logger
from monai.engines import SupervisedTrainer,SupervisedEvaluator
from monai.data import DataLoader
from monai.networks import eval_mode
from monai.networks.nets import UNet, densenet121
from monai.inferers import SlidingWindowInferer,SimpleInferer
from monai.handlers import (
    CheckpointLoader,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    from_engine
)
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    EnsureTyped,
    SaveImaged,
)

from dataset import clfDataset,segDataset

def eval_clf2d(config:dict):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("eval_log")
    # Get config parameters
    device = torch.device(config['device'])
    root_dir = config['root_dir']
    model_name = config['model_name']
    image_size = config['image_size']
    model_file = config['model_file']
    if model_name == 'densenet':
        model = densenet121(
                spatial_dims=2, 
                in_channels=3, 
                out_channels=4).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")


    dataset_dir = Path(root_dir)
    class_names = sorted(f"{x.name}" for x in dataset_dir.iterdir() if x.is_dir())

    val_ds = clfDataset.from_default(root=root_dir,is_train=False,size=image_size)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)


    val_post_transforms = Compose(
    [
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
        KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3]),
        SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=config["output_dir"])
        ]
    )

    val_handlers = [
        StatsHandler(name="eval_log", output_transform=lambda x: None),
        CheckpointLoader(load_path=model_file, load_dict={"model":model}),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        inferer=SlidingWindowInferer(roi_size=(64,64,64), sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (from_engine(["pred", "label"])(x)[0], [monai.networks.utils.one_hot(xx, dim=0, num_classes=4) for xx in from_engine(["pred", "label"])(x)[1]]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (from_engine(["pred", "label"])(x)[0], [monai.networks.utils.one_hot(xx, dim=0, num_classes=4) for xx in from_engine(["pred", "label"])(x)[1]]))},
        val_handlers=val_handlers,
    )

    evaluator.run()

    max_items_to_print = 10
    with eval_mode(model):
        for item in DataLoader(testdata, batch_size=1, num_workers=0):
            prob = np.array(model(item["image"].to(device))).detach().to("cpu"))[0]
            pred = class_names[prob.argmax()]
            gt = item["class_name"][0]
            print(f"Class prediction is {pred}. Ground-truth: {gt}")
            max_items_to_print -= 1
            if max_items_to_print == 0:
                break




def eval_seg3d(config:dict):
    # Set up logging
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("eval_log")

    # Get config parameters
    device = torch.device(config['device'])
    model_name = config['model_name']
    model_file = config['model_file']
    # Initialize model
    if model_name == 'unet3d':
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        val_img_path = config['val_img_path']
        val_seg_path = config['val_seg_path']
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Create datasets

    val_ds = segDataset.from_default(
        image_dir=val_img_path,
        seg_dir=val_seg_path,
        is_train=False,
    )

    # Create data loaders
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # Set up validation transforms and handlers
    val_post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3]),
            SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=config["output_dir"])
        ]
    )

    val_handlers = [
        StatsHandler(name="eval_log", output_transform=lambda x: None),
        CheckpointLoader(load_path=model_file, load_dict={"model":model}),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        inferer=SlidingWindowInferer(roi_size=(64,64,64), sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (from_engine(["pred", "label"])(x)[0], [monai.networks.utils.one_hot(xx, dim=0, num_classes=4) for xx in from_engine(["pred", "label"])(x)[1]]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (from_engine(["pred", "label"])(x)[0], [monai.networks.utils.one_hot(xx, dim=0, num_classes=4) for xx in from_engine(["pred", "label"])(x)[1]]))},
        val_handlers=val_handlers,
    )
    evaluator.run()


if __name__ == "__main__":
    config = {
        "device": "cpu",
        "model_name": "unet3d",
        "output_dir": "./output/",
        "model_file": "./checkpoints/checkpoint_epoch=10.pt",
        "val_img_path": "./data/brats_val/imageTr",
        "val_seg_path": "./data/brats_val/labelTr",
    }
    eval_seg3d(config)
