import logging
import sys
import os
import monai
import torch
import mlflow
from ignite.metrics import Accuracy, Loss

from monai.apps import get_logger
from monai.engines import SupervisedTrainer,SupervisedEvaluator
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.networks.nets import UNet,densenet121,UNETR
from monai.inferers import SlidingWindowInferer,SimpleInferer
from monai.handlers import (
    CheckpointSaver,
    EarlyStopHandler,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    from_engine,
    MLFlowHandler
)
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    EnsureTyped,
)
from monai.utils import set_determinism
from ignitetls.dataset import clfDataset,segDataset


def train_clf2d(cfg:dict):
    set_determinism(seed=cfg['seed'])
    # Set up logging
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("train_log")

    # Get config parameters
    device = torch.device(cfg['device'])
    max_epochs = cfg['max_epochs']
    model_name = cfg['model_name']
    image_size = cfg['image_size']
    batch_size = cfg['batch_size']

    if model_name == 'densenet':
        model = densenet121(
                spatial_dims=2, 
                in_channels=3, 
                out_channels=4).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    train_ds = clfDataset.from_default(
        root=cfg['train_img_path'],
        is_train=True, 
        size=image_size,
    )
    val_ds = clfDataset.from_default(
        root=cfg['val_img_path'],
        is_train=False,
        size=image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    # Initialize training components
    loss_fun =torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = opt,
        mode='min',
        factor=0.5,
        patience=10)


    val_handlers = [
        CheckpointSaver(
            save_dir=cfg['save_dir'], 
            save_dict={"model":model}, 
            save_key_metric=True,
            save_final=True,
            final_filename="clf2d_best_model.pt"),
        StatsHandler(output_transform=lambda x: None),

    ]


    val_postpreprocessing = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        inferer=SimpleInferer() ,
        postprocessing=val_postpreprocessing,
        key_val_metric={"val_acc": Accuracy(output_transform=from_engine(["pred","label"])),},
        additional_metrics ={"val_loss":Loss(loss_fun,output_transform=from_engine(["pred","label"]))},
        val_handlers=val_handlers,
    )


    train_postprocessing = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )

    train_handlers = [

        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True,step_transform=lambda x:x.state.output[0]["loss"]),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        CheckpointSaver(save_dir=cfg['save_dir'], save_dict={"model":model, "opt":opt}, save_interval=2, epoch_level=True),
        # MLFlowHandler(tracking_uri=cfg['mlflow_uri'], experiment_name=cfg['experiment_name'],output_transform=from_engine(["loss"],first=True)),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=opt,
        loss_function=loss_fun,
        inferer=SimpleInferer(),
        postprocessing=train_postprocessing,
        train_handlers=train_handlers,
    )

    trainer.run()


def train_seg3d(cfg:dict):
    set_determinism(seed=cfg['seed'])
    # Set up logging
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("train_log")

    # Get config parameters
    device = torch.device(cfg['device'])
    max_epochs = cfg['max_epochs']
    model_name = cfg['model_name']
    chunk_size = cfg['chunk_size']
    train_img_path = cfg['train_img_path']
    val_img_path = cfg['val_img_path']
    train_seg_path = cfg['train_seg_path']
    val_seg_path = cfg['val_seg_path']
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

    elif model_name == 'unetr':
        model = UNETR(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            img_size=(64, 64, 64),
            feature_size=16,
        ).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Create datasets
    train_ds = segDataset.from_default(
        image_dir=train_img_path,
        seg_dir=train_seg_path,
        is_train=True,
        chunck_size=chunk_size
    )

    val_ds = segDataset.from_default(
        image_dir=val_img_path,
        seg_dir=val_seg_path,
        is_train=False,
    )

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # Initialize training components
    loss = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    # Set up validation transforms and handlers
    val_post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3]),
        ]
    )

    val_handlers = [
        EarlyStopHandler(trainer=None, patience=2, score_function=lambda x:x.state.metrics["val_mean_dice"]),
        StatsHandler(name="train_log", output_transform=lambda x: None),
        CheckpointSaver(save_dir=cfg['save_dir'], save_dict={"model":model}, save_key_metric=True),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        inferer=SlidingWindowInferer(roi_size=chunk_size, sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (from_engine(["pred", "label"])(x)[0], [monai.networks.utils.one_hot(xx, dim=0, num_classes=4) for xx in from_engine(["pred", "label"])(x)[1]]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (from_engine(["pred", "label"])(x)[0], [monai.networks.utils.one_hot(xx, dim=0, num_classes=4) for xx in from_engine(["pred", "label"])(x)[1]]))},
        val_handlers=val_handlers,
    )

    # # Set up training transforms and handlers
    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3]),
        ]
    )

    train_handlers = [
        EarlyStopHandler(trainer=None, patience=20, score_function=lambda x:-x.state.output[0]["loss"], epoch_level=False),
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=2, epoch_level=True),
        StatsHandler(name="train_log", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        CheckpointSaver(save_dir=cfg['save_dir'], save_dict={"model":model, "opt":opt}, save_interval=2, epoch_level=True),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        network=model,
        train_data_loader=train_loader,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=train_post_transforms,
        key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (from_engine(["pred", "label"])(x)[0], [monai.networks.utils.one_hot(xx, dim=0, num_classes=4) for xx in from_engine(["pred", "label"])(x)[1]]))},
        train_handlers=train_handlers,
    )

    # Link handlers and start training
    val_handlers[0].set_trainer(trainer=trainer)
    train_handlers[0].set_trainer(trainer=trainer)
    trainer.run()


if __name__ == "__main__":
    # cfg = {
    #     "seed": 42,
    #     "device": "cpu",
    #     "max_epochs": 5,
    #     "model_name": "unetr",
    #     "save_dir": "./checkpoints",
    #     "chunk_size": (64, 64, 64),
    #     "learning_rate": 0.001,
    #     "train_img_path": "./data/brats_train/imageTr",
    #     "train_seg_path": "./data/brats_train/labelTr",
    #     "val_img_path": "./data/brats_val/imageTr",
    #     "val_seg_path": "./data/brats_val/labelTr",
    # }
    # train_seg3d(cfg)

    cfg = {
        "seed": 42,
        "device": "cpu",
        "max_epochs": 2,
        "model_name": "densenet",
        "image_size": (256, 256),
        "learning_rate": 1e-5,
        "save_dir": "./checkpoints",
        "batch_size": 32,
        "train_img_path": "./data/raw_data/brain-tumor-classification-mri/Testing",
        "val_img_path": "./data/raw_data/brain-tumor-classification-mri/Testing",
        "mlflow_uri": "http://localhost:5000",
        "experiment_name": "brain"
    }
    train_clf2d(cfg)
