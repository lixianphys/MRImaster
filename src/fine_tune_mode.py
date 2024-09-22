from metaflow import FlowSpec, IncludeFile, Parameter, step
import pandas as pd
import pathlib
import splitfolders
import torch
import torchvision
import torchvision.transforms as transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch import optim
from PIL import Image
from utils import script_path
from network import default_params_model
import wandb
import os
from datapipe import KaggleDataPipe

kaggle_link = "sartajbhuvaji/brain-tumor-classification-mri"
dir_to_store = "data/raw_data/brain-tumor-classification-mri/"

brain_tumor_dt = KaggleDataPipe(kaggle_link,dir_to_store)
brain_tumor_dt.load_from_kaggle()
labels = brain_tumor_dt.get_labels("Training")

DATASET = os.path.join(dir_to_store,"Training")
OUTPUT = "data/processed_data/brain-tumor-classification-mri"

class FineTuneDropoutRate(FlowSpec):
    """
    Fine tune hyperparameters, such as dropout rate
    """
    mode = Parameter(
        name = "mode",
        help='Determines if running test on a small dataset', 
        default="small")

    model_params = Parameter(
        name = "model_params",
        help="The parameters for the model.",
        default=default_params_model,
    )

    image_path = Parameter(
        name = "image_path",
        help="The path to the image file.",
        default="pred_examples/not cancer.jpg",
    )

    load_params = Parameter(
        name = "load_params",
        help="The parameters for loading the data.",
        default={"train_ratio": 0.8, "batch_size": 64, "image_size":(256,256)},
    )

    model_save_path = Parameter(
        name = "save_model_to_path",
        help="The path to save the parameters of the best model.",
        default="model_to_deploy.pt",
    )

    @step
    def start(self):
        """
        The start step:
        1) Loads labels into pandas dataframe.
        2) Split the dataset into train and test set.
        """
        # Dataset Path
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"))
        
        data_dir = pathlib.Path(DATASET)
        train_ratio = self.load_params["train_ratio"]

        wandb.config.train_ratio=train_ratio

        splitfolders.ratio(data_dir, output=OUTPUT, seed=20, ratio=(train_ratio, 1-train_ratio))
        # new dataset path
        data_dir = pathlib.Path(OUTPUT)

        # define transformation
        transform = transforms.Compose(
            [
                transforms.Resize((256,256)),
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

        # import and load train, validation
        batch_size = self.load_params["batch_size"]

        if self.mode == "small":
        
            batch_size = 25
            train_set = torch.utils.data.Subset(train_set, torch.arange(0, 500))
            val_set = torch.utils.data.Subset(val_set, torch.arange(0, 500))

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 2)
        self.dropout_rate = [0.1,0.15,0.2,0.25]
        self.next(self.build_model, foreach="dropout_rate")
    
    @step
    def build_model(self):

        from network import CNN_TUMOR
        from torchsummary import summary
        # Assign the dropout rate to each branch
        self.model_params["dropout_rate"] = self.input
        self.model = CNN_TUMOR(self.model_params)
        # define computation hardware approach (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(summary(self.model, input_size = self.model_params["shape_in"],device=self.device.type))
        self.next(self.train)
    
    @step
    def train(self):
        """
        Train on train and evaluate on eval
        """
        from utils import get_lr, loss_batch, loss_epoch
        import copy
        from tqdm import tqdm

        self.train_params = {
            "train": self.train_loader,
            "val": self.val_loader,
            "epochs": 20,
            "lr"    : 3e-4,
            "optimiser": optim.Adam(self.model.parameters(),lr=3e-4),
            "lr_change": ReduceLROnPlateau(optim.Adam(self.model.parameters(),
                                            lr=3e-4),
                                            mode='min',
                                            factor=0.5,
                                            patience=20,
                                            verbose=0),
            "f_loss": nn.NLLLoss(reduction="sum"),
            "weight_path": "model.pt",
        }
  
        # define loss function and optimizer
        loss_func = nn.NLLLoss(reduction="sum")
        opt = optim.Adam(self.model.parameters(), lr=self.train_params["lr"])
        lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)
        self.model = self.model.to(self.device)

        # Get the parameters
        epochs=self.train_params["epochs"]

        loss_func=self.train_params["f_loss"]
        opt=self.train_params["optimiser"]
        train_dl=self.train_params["train"]
        val_dl=self.train_params["val"]
        lr_scheduler=self.train_params["lr_change"]
        weight_path=self.train_params["weight_path"]
        
        # history of loss values in each epoch
        loss_history={"train": [],"val": []} 
        # histroy of metric values in each epoch
        metric_history={"train": [],"val": []} 
        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(self.model.state_dict()) 
        # initialize best loss to a large value
        best_loss=float('inf') 

        # Train Model n_epochs (the progress of training by printing the epoch number and the associated learning rate. It can be helpful for debugging, monitoring the learning rate schedule, or gaining insights into the training process.) 
    
        for epoch in tqdm(range(1, epochs)):
            
            # Get the Learning Rate
            current_lr=get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs, current_lr))

            # Train Model Process
            self.model.train()
            train_loss, train_metric = loss_epoch(self.model,self.device,loss_func,train_dl,opt)

            # collect losses
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)


            # Evaluate Model Process
            self.model.eval()
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(self.model,self.device, loss_func,val_dl)
            
            # store best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                
                # store weights into a local file
                torch.save(self.model.state_dict(), weight_path)
                print("Copied best model weights!")
            
            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)
            
            # learning rate schedule
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                self.model.load_state_dict(best_model_wts) 

                print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
                print("-"*10) 

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.loss_history = loss_history 
        self.metric_history = metric_history
        self.next(self.join)
    
    @step
    def join(self,inputs):
        """
        Join the results of the training.
        """
        self.loss_history = [input.loss_history for input in inputs]
        self.metric_history = [input.metric_history for input in inputs]
        self.next(self.end)
        
    
    @step
    def end(self):
        """
        End the flow.
        """
        pass        

if __name__ == "__main__":
    FineTuneDropoutRate()


