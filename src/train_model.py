""" 
This is the main file to train the model.
"""
import pandas as pd
import pathlib
import splitfolders
import torch
import torchvision
import torchvision.transforms as transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import os
from torch import optim
from PIL import Image
from datapipe import KaggleDataPipe


kaggle_link = "sartajbhuvaji/brain-tumor-classification-mri"
dir_to_store = "data/raw_data/brain-tumor-classification-mri/"

brain_tumor_dt = KaggleDataPipe(kaggle_link,dir_to_store)
brain_tumor_dt.load_from_kaggle()
labels = brain_tumor_dt.get_labels("Training")

DATASET = os.path.join(dir_to_store,"Training")
OUTPUT = "data/processed_data/brain-tumor-classification-mri"

class TumorAnalysisModel(object):
    """
    The workflow performs the following steps:
    1) Ingest a CSV into a Pandas Dataframe and split it into a train, eval, and test split
    2) Create the train_set and val_set
    3) Train the model
    """
    def __init__(self, mode="small", verbose=False):
        self.mode = mode
        self.verbose = verbose

    def load_data(self, load_params):
        """
        The start step:
        1) Loads labels into pandas dataframe.
        2) Split the dataset into train and test set.
        """
        if self.verbose:
            print("Loading Data")
        # Dataset Path
        data_dir = pathlib.Path(DATASET)
        train_ratio = load_params["train_ratio"]
        splitfolders.ratio(data_dir, output=OUTPUT, seed=20, ratio=(train_ratio, 1-train_ratio))
        # new dataset path
        data_dir = pathlib.Path(OUTPUT)

        # define transformation
        transform = transforms.Compose(
            [
                transforms.Resize(load_params["image_size"]),
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
        batch_size = load_params["batch_size"]

        if self.mode == "small":
        
            batch_size = 10
            train_set = torch.utils.data.Subset(train_set, torch.arange(0, 100))
            val_set = torch.utils.data.Subset(val_set, torch.arange(0, 100))

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 2)
    
    def build_model(self, model_params):
        if self.verbose:
            print("\n Building Model")

        from network import CNN_TUMOR
        from torchsummary import summary
        self.model = CNN_TUMOR(model_params)
        # define computation hardware approach (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(summary(self.model, input_size= model_params["shape_in"],device=self.device.type))

    def train(self, train_params,verbose=False):
        """
        Train on train and evaluate on eval
        """
        from utils import get_lr, loss_batch, loss_epoch
        import copy
        from tqdm import tqdm
        if self.verbose:
            print("\n Training Model")
        # define loss function and optimizer
        loss_func = nn.NLLLoss(reduction="sum")
        opt = optim.Adam(self.model.parameters(), lr=train_params["lr"])
        lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)
        self.model = self.model.to(self.device)

        # Get the parameters
        epochs=train_params["epochs"]

        loss_func=train_params["f_loss"]
        opt=train_params["optimiser"]
        train_dl=train_params["train"]
        val_dl=train_params["val"]
        lr_scheduler=train_params["lr_change"]
        weight_path=train_params["weight_path"]
        
        # history of loss values in each epoch
        loss_history={"train": [],"val": []} 
        # histroy of metric values in each epoch
        metric_history={"train": [],"val": []} 
        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(self.model.state_dict()) 
        # initialize best loss to a large value
        best_loss=float('inf') 

        # Train Model n_epochs (the progress of training by printing the epoch number and the associated learning rate. It can be helpful for debugging, monitoring the learning rate schedule, or gaining insights into the training process.) 
    
        for epoch in tqdm(range(epochs)):
            
            # Get the Learning Rate
            current_lr=get_lr(opt)
            if verbose:
                print('Epoch {}/{}, current lr={}'.format(epoch+1, epochs, current_lr))

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
                self.model.load_state_dict(best_model_wts) 

            if verbose:
                print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
                print("-"*10) 

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        self.train_params = train_params
        self.loss_history = loss_history 
        self.metric_history = metric_history

    def metrics_visualisation(self):
        """
        Visualize the metrics
        """
        if self.verbose:
            print("\n Visualizing Metrics")
        # Convergence History Plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        epochs=self.train_params["epochs"]
        fig,ax = plt.subplots(1,2,figsize=(12,5))

        sns.lineplot(x=[*range(1,epochs+1)],y=
                     self.loss_history["train"],ax=ax[0],label='loss_hist["train"]')
        sns.lineplot(x=[*range(1,epochs+1)],y=
                     self.loss_history["val"],ax=ax[0],
                     label='loss_hist["val"]')
        sns.lineplot(x=[*range(1,epochs+1)],y=
                     self.metric_history["train"],ax=ax[1],label='Acc_hist["train"]')
        sns.lineplot(x=[*range(1,epochs+1)],y=
                     self.metric_history["val"],ax=ax[1],
                     label='Acc_hist["val"]')
        plt.show()

    def predict(self,image):
        if self.verbose:
            print("\n Predicting")
        from utils import preprocess_image
        # Preprocess the image
        image = preprocess_image(image)
        
        # Run the model on the input image
        with torch.no_grad():  # Disable gradient computation for faster inference
            output = self.model(image)
        
        # Optionally, apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get the predicted class (assuming single-label classification)
        _, predicted_class = torch.max(probabilities, 1)
        
        return predicted_class.item(), probabilities

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path, model_params):
        from network import CNN_TUMOR
        self.model = CNN_TUMOR(model_params)
        self.model.load_state_dict(torch.load(path, weights_only=True))

if __name__ == "__main__":
    from utils import CLA_label

    workflow = TumorAnalysisModel("small", verbose=True)
    
    # Load the data
    load_params = {
        "train_ratio": 0.8,
        "batch_size": 64,
        "image_size": (256,256),
    }
    workflow.load_data(load_params)

    model_params={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 4}
    workflow.build_model(model_params)

    train_params = {
        "train": workflow.train_loader,
        "val": workflow.val_loader,
        "epochs": 5,
        "lr"    : 3e-4,
        "optimiser": optim.Adam(workflow.model.parameters(),lr=3e-4),
        "lr_change": ReduceLROnPlateau(optim.Adam(workflow.model.parameters(),
                                        lr=3e-4),
                                        mode='min',
                                        factor=0.5,
                                        patience=20,
                                        verbose=0),
        "f_loss": nn.NLLLoss(reduction="sum"),
        "weight_path": "model.pt",
    }
    workflow.train(train_params, verbose=True)
    workflow.metrics_visualisation() 
    # prediction, probability = workflow.predict(Image.open("pred_examples/not cancer.jpg").convert('RGB'))
    # print(CLA_label[prediction], probability.numpy()[0][prediction])










