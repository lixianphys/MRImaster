import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pathlib
import splitfolders
import mlflow
from tqdm import tqdm 
from src.preprocess.nifti import LazyLoadingNiftiDataset
from torch.utils.data import DataLoader
from src.unet3d import UNet3D
from src.cnn import CNN_TUMOR
from src.utils.utils import get_lr,loss_epoch
import copy


def train_cnn(config):
    """ train a simple CNN model"""

    shape_in = tuple(config['model']['shape_in'])
    num_classes = config['model']['num_classes']
    initial_filters = config['model']['initial_filters']
    num_fc1 = config['model']['num_fc1']
    dropout_rate = config['model']['dropout_rate']
    image_size = tuple(config['loading']['image_size'])
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    verbose = config['training']['verbose']
    learning_rate = float(config['training']['learning_rate'])
    mlflow_enabled = config['mlflow']['enabled']
    model_save_path = config['model']['save_path']
    device=torch.device(config['training']['device'])

    model = CNN_TUMOR(
        {
        'shape_in':shape_in,
        'num_classes':num_classes,
        'initial_filters': initial_filters,
        'num_fc1':num_fc1,
        'dropout_rate':dropout_rate
    }
    )
    print("Data Augmentation Starts ...")
    data_dir = pathlib.Path(config['data']['dataset_path'])
    train_ratio = config['loading']['train_ratio']
    splitfolders.ratio(data_dir, output=config['data']['output_path'], seed=20, ratio=(train_ratio, 1-train_ratio))
    # new dataset path
    data_dir = pathlib.Path(config['data']['output_path'])

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
    train_set = torchvision.datasets.ImageFolder(data_dir.joinpath("train"), transform=transform) 
    val_set = torchvision.datasets.ImageFolder(data_dir.joinpath("val"), transform=transform)

    train_dl = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
    val_dl = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 2)
    loss_func = nn.NLLLoss(reduction="sum")
    opt = optim.Adam(model.parameters(),lr=learning_rate) 
    lr_scheduler =  ReduceLROnPlateau(
        optimizer =optim.Adam(model.parameters(),
        lr=learning_rate),
        mode='min',
        factor=0.5,
        patience=20)
    
    if verbose:
        print("\n Training start...")
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
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
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
        model_save_path = config['model']['save_path']
        torch.save(model.state_dict(), model_save_path)
        mlflow.log_artifact(model_save_path, artifact_path="models")
        mlflow.end_run()

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        # Cross-entropy loss
        ce_loss = self.cross_entropy(outputs, labels)
        
        # Dice loss for each class
        outputs_soft = F.softmax(outputs, dim=1)  # Softmax along the channel dimension
        labels_one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        dice_loss_value = DiceCrossEntropyLoss.dice_loss(outputs_soft, labels_one_hot)
        
       # Combine losses (equal weighting of CE and Dice)
        total_loss = ce_loss + dice_loss_value
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

def train_unet(config):
    """ Train a 3D Unet model"""
    in_channels=config['model']['in_channels']
    out_channels=config['model']['out_channels']
    epochs=config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    verbose = config['training']['verbose']
    mlflow_enabled = config['mlflow']['enabled']

    # MLflow logging setup
    if mlflow_enabled:
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        mlflow.start_run()
        
        # Log model hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("in_channels",in_channels)
        mlflow.log_param("out_channels",out_channels)

    # Paths to NIfTI images and labels
    brats_data_path = config['data']['data_path']
    # place all images (nii or nii.gz) in data_path/imageTr
    img_folder = os.path.join(brats_data_path,'imageTr')
    # place all labels (nii or nii.gz) in data_path/labelTr
    lbl_folder = os.path.join(brats_data_path,'labelTr')
    img_filenames = os.listdir(img_folder)
    lbl_filenames = os.listdir(lbl_folder)


    image_paths = [os.path.join(img_folder,file_path) for file_path in img_filenames]
    label_paths = [os.path.join(lbl_folder,file_path) for file_path in lbl_filenames]
    cache_dir = config['data']['cache_path']
    # Initialize the dataset with caching
    dataset = LazyLoadingNiftiDataset(image_paths=image_paths, label_paths=label_paths, cache_dir=cache_dir)

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device  = torch.device(torch.load(config['training']['device']))
    model = UNet3D(in_channels,out_channels)
    model = model.to(device)

    criterion = DiceCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Example training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for batch, (inputs, labels) in enumerate(dataloader, desc="Training"):  # inputs shape: (batch_size, 4, H, W, D), labels shape: (batch_size, H, W, D)
            optimizer.zero_grad()
            
            outputs = model(inputs)  # outputs shape: (batch_size, num_classes, H, W, D)
            loss = criterion(outputs, labels)  # Compute the combined Dice + Cross-Entropy loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if verbose:
                print(f"Batch {batch+1}/{len(dataloader)}, Loss: {loss.item()}")
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
                # Log metrics to MLflow
        if mlflow_enabled:
            mlflow.log_metric("train_loss", running_loss/len(dataloader), step=epoch)
        # Log model artifact
    if mlflow_enabled:
        model_save_path = config['model']['save_path']
        torch.save(model.state_dict(), model_save_path)
        mlflow.log_artifact(model_save_path, artifact_path="models")
        mlflow.end_run()

   
