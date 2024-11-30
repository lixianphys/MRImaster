import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
import pathlib
import splitfolders
import mlflow
from tqdm import tqdm 
from abc import ABC, abstractmethod

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


from src.preprocess.nifti import LazyLoadingNiftiDataset,NormalLoadingNiftiDataset
from src.unet3d import UNet3D
from src.cnn import CNN_TUMOR
from src.utils.utils import (
    get_lr,
    loss_epoch,
    iou_per_class,
    dice_score_per_class,
    check_and_prepare_file_path,
    delete_folder_without_confirmation)
from src.utils.configYaml import (
    pretty_print_config,
    extract_hyperparameters,
    string_tuple_to_tuple)
import copy
from typing import Dict, Union

cnn_keys_to_print = [
    'model',
    'train.load.train_ratio',
    'train.batch_size',
    'train.epochs',
    'train.learning_rate']



class Train(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class TrainCNN(Train):
    """ train a simple CNN model"""
    def __init__(self):
        self.counts = 0
    def __call__(self,config:Dict):
        self.counts += 1 # keep track of number of training in batch.
            
        print("Training with configuration:\n")
        pretty_print_config(extract_hyperparameters(
            config, extracted_keys = cnn_keys_to_print)
        )
        shape_in = string_tuple_to_tuple(config['model']['shape_in'])
        num_classes = config['model']['num_classes']
        initial_filters = config['model']['initial_filters']
        num_fc1 = config['model']['num_fc1']
        dropout_rate = config['model']['dropout_rate']
        image_size = string_tuple_to_tuple(config['train']['load']['image_size'])
        skip_loading = config['train']['skip_loading']
        batch_size = config['train']['batch_size']
        epochs = config['train']['epochs']
        verbose = config['train']['verbose']
        train_ratio = config['train']['load']['train_ratio']
        learning_rate = float(config['train']['learning_rate'])
        mlflow_enabled = config['train']['mlflow']['enabled']
        model_save_path = check_and_prepare_file_path(
            os.path.join(
                config['train']['save_path'],
                f'Exp_{self.counts}.pt' if config['train']['checkpoint_name']=="#" else config['train']['checkpoint_name']
            )
        )

        device=torch.device(config['train']['device'])

        model = CNN_TUMOR(
            {
            'shape_in':shape_in,
            'num_classes':num_classes,
            'initial_filters': initial_filters,
            'num_fc1':num_fc1,
            'dropout_rate':dropout_rate
            }
        )
        if not skip_loading: # load and augment data
            print(f"Loading Data into Train and Validation folder with the train_ratio {train_ratio}...")
            data_dir = pathlib.Path(config['train']['data']['dataset'])
            splitfolders.ratio(data_dir, output=config['train']['data']['output'], seed=20, ratio=(train_ratio, 1-train_ratio))
            # new dataset path
        
        data_dir = pathlib.Path(config['train']['data']['output'])

        print("Data Augmentation Starts ...")

        # define transformation
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ]
        )    
        # Define an object of the custom dataset for the train and validation.
        train_set = torchvision.datasets.ImageFolder(data_dir.joinpath(config['train']['data']['train_set']), transform=transform) 
        val_set = torchvision.datasets.ImageFolder(data_dir.joinpath(config['train']['data']['val_set']), transform=transform)

        train_dl = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
        val_dl = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 2)

        loss_func = nn.NLLLoss(reduction="sum")
        opt = optim.Adam(model.parameters(),lr=learning_rate) 
        lr_scheduler =  ReduceLROnPlateau(
            optimizer = opt,
            mode='min',
            factor=0.5,
            patience=10)
        
        if verbose:
            print("Training start...")
        # define loss function and optimizer

        model = model.to(device)
        
        # history of loss values in each epoch
        loss_history={"train": [],"val": []} 
        # histroy of metric values in each epoch
        metric_history={"train": [],"val": []} 
        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(model.state_dict()) 
        # initialize best loss to a large value
        best_loss=float('inf') 

        # Train Model n_epochs (the progress of training by printing the epoch number and the associated learning rate. It can be helpful for debugging, monitoring the learning rate schedule, or gaining insights into the training process.) 

        # MLflow logging setup
        if mlflow_enabled:
            mlflow.set_tracking_uri(config['train']['mlflow']['uri'])
            mlflow.set_experiment(config['train']['mlflow']['experiment'])
            mlflow.start_run()
            
            # Log model hyperparameters
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("optimizer", "adam")
            mlflow.log_param("shape_in",shape_in)
            mlflow.log_param("initial_filters",initial_filters)
            mlflow.log_param("image_size",image_size)
            mlflow.log_param("num_classes",num_classes)
            mlflow.log_param("train_ratio",train_ratio)
            mlflow.log_param("num_fc1",num_fc1)


        for epoch in tqdm(range(epochs)):
            
            # Get the Learning Rate
            current_lr=get_lr(opt)
            if verbose:
                print('Epoch {}/{}, current lr={}'.format(epoch+1, epochs, current_lr))

            # Train Model Process
            model.train()
            train_loss, train_metric = loss_epoch(model,device,loss_func,train_dl,opt)

            # collect losses
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)


            # Evaluate Model Process
            model.eval()
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(model,device, loss_func,val_dl)
            
            # store best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # store weights into a local file
                torch.save(model.state_dict(), model_save_path)
                if verbose:
                    print("Copied best model weights!")
            
            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)
            
            # learning rate schedule
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                if verbose:
                    print("Loading best model weights!")
                model.load_state_dict(best_model_wts) 

            if verbose:
                print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
                print("-"*10)

            # Log metrics to MLflow
            if mlflow_enabled:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_metric", train_metric, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_metric", val_metric, step=epoch)

        # Log model artifact
        if mlflow_enabled:
            torch.save(model.state_dict(), model_save_path)
            mlflow.log_artifact(model_save_path, artifact_path="models")
            mlflow.end_run()

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, celoss_ratio:Union[float,int] =1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.celoss_ratio = celoss_ratio

    def forward(self, outputs, labels):
        # Cross-entropy loss
        ce_loss = self.cross_entropy(outputs, labels)
        
        # Dice loss for each class
        outputs_soft = F.softmax(outputs, dim=1)  # Softmax along the channel dimension
        labels_one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        dice_loss_value = DiceCrossEntropyLoss.dice_loss(outputs_soft, labels_one_hot)
        
       # Combine losses (equal weighting of CE and Dice)
        total_loss = ce_loss*self.celoss_ratio + dice_loss_value
        return total_loss
    
    @staticmethod
    def dice_loss(outputs_soft, labels_one_hot, smooth=1.0):
        """
        Compute the average Dice loss across all classes.
        """
        dice_loss_per_class = []
        for i in range(outputs_soft.shape[1]):  # Loop over each class
            pred = outputs_soft[:, i]  # Softmax probability for class `i`
            target = labels_one_hot[:, i]  # One-hot encoded ground truth for class `i`
            intersection = (pred * target).sum(dim=(1, 2, 3))  # Sum over spatial dimensions
            dice = (2.0 * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
            dice_loss_per_class.append(1 - dice)  # Dice loss for class `i`
        # Average Dice loss over batch and classes
        dice_loss = torch.mean(torch.stack(dice_loss_per_class, dim=1), dim=1)  # Average over classes
        return dice_loss.mean()  # Average over batch

unet_keys_to_print = [
    'model',
    'train.batch_size',
    'train.epochs',
    'train.learning_rate']

class TrainUnet(Train):
    """ Train a 3D Unet model"""
    def __init__(self):
        self.counts = 0
    def __call__(self,config:Dict):
        self.counts+=1
        print("Training with configuration:\n")
        pretty_print_config(extract_hyperparameters(
            config, extracted_keys = unet_keys_to_print)
        )        
        in_channels=config['model']['in_channels']
        out_channels=config['model']['out_channels']
        epochs=config['train']['epochs']
        batch_size = config['train']['batch_size']
        learning_rate = config['train']['learning_rate']
        verbose = config['train']['verbose']
        celoss_ratio = float(config['train']['celoss_ratio'])
        mlflow_enabled = config['train']['mlflow']['enabled']
        image_folder = config['train']['data']['image_folder']
        label_folder = config['train']['data']['label_folder']
        model_save_path = check_and_prepare_file_path(
            os.path.join(
                config['train']['save_path'],
                f'Exp_{self.counts}.pt' if config['train']['checkpoint_name']=="#" else config['train']['checkpoint_name']
            )
        )

        # MLflow logging setup
        if mlflow_enabled:
            mlflow.set_tracking_uri(config['train']['mlflow']['uri'])
            mlflow.set_experiment(config['train']['mlflow']['experiment'])
            mlflow.start_run()
            
            # Log model hyperparameters
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("optimizer", "adam")
            mlflow.log_param("in_channels",in_channels)
            mlflow.log_param("out_channels",out_channels)

        # Paths to NIfTI images and labels
        brats_data_path = config['train']['data']['data_path']
        # place all images (nii or nii.gz) in data_path/image_folder
        img_folder = os.path.join(brats_data_path,image_folder)
        # place all labels (nii or nii.gz) in data_path/label_folder
        lbl_folder = os.path.join(brats_data_path,label_folder)
        img_filenames = os.listdir(img_folder)
        lbl_filenames = os.listdir(lbl_folder)


        image_paths = [os.path.join(img_folder,file_path) for file_path in img_filenames]
        label_paths = [os.path.join(lbl_folder,file_path) for file_path in lbl_filenames]
        cache_dir = config['train']['data']['cache_path']
        # Initialize the dataset with caching
        dataset = NormalLoadingNiftiDataset(image_paths=image_paths, label_paths=label_paths)

        # Create a DataLoader for batching
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        device  = torch.device(config['train']['device'])
        model = UNet3D(in_channels,out_channels)
        model = model.to(device)

        criterion = DiceCrossEntropyLoss(celoss_ratio)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        lr_scheduler =  ReduceLROnPlateau(
            optimizer = optimizer,
            mode='min',
            factor=0.5,
            patience=10)

        # Example training loop
        for epoch in tqdm(range(epochs)):
            model.train()
            running_loss = 0.0
            dice_score = [0.0]*out_channels
            iou_score = [0.0]*out_channels
            for batch, (inputs, labels) in enumerate(dataloader):  # inputs shape: (batch_size, 4, H, W, D), labels shape: (batch_size, H, W, D)
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)  # outputs shape: (batch_size, num_classes, H, W, D)
                loss = criterion(outputs, labels)  # Compute the combined Dice + Cross-Entropy loss
                for c in range(out_channels):
                    dice_class = dice_score_per_class(torch.argmax(outputs, dim=1), labels, num_classes=out_channels)
                    dice_score[c] += dice_class[c]
                    iou_class = iou_per_class(torch.argmax(outputs, dim=1), labels, num_classes=out_channels)
                    iou_score[c]  += iou_class[c]
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if verbose:
                    print(f"Batch {batch+1}/{len(dataloader)}, Loss: {loss.item()}")
                    
            # learning rate schedule
            lr_scheduler.step(running_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
                print(f"Last Learning Rate is {lr_scheduler.get_last_lr()[0]}")
            
            # Log metrics to MLflow
            if mlflow_enabled:
                mlflow.log_metric("train_loss", running_loss/len(dataloader), step=epoch)
                for cls in range(out_channels):
                    mlflow.log_metric(f"iou_score_class{cls}",iou_score[cls]/len(dataloader), step=epoch)
                    mlflow.log_metric(f"dice_score_class{cls}", dice_score[cls]/len(dataloader), step=epoch)
            # Log model artifact
        if mlflow_enabled:
            torch.save(model.state_dict(), model_save_path)
            mlflow.log_artifact(model_save_path, artifact_path="models")
            mlflow.end_run()

   
