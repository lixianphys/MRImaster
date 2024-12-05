import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import cv2
import lightning as L


# Define Architecture For CNN_TUMOR Model
class LiCNN(L.LightningModule):
    # Network Initialisation
    def __init__(self, params):
        super().__init__()
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        self.learning_rate = params.get("learning_rate", 1e-3)
    
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            self.convBlock(Cin, init_f),
            self.convBlock(init_f, 2 * init_f),
            self.convBlock(2 * init_f, 4 * init_f),
            self.convBlock(4 * init_f, 8 * init_f)
        )

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, Cin, Hin, Win)
            flatten_dim = self.conv_layers(dummy_input).numel()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, num_fc1),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_fc1, num_classes)
        )

    def forward(self,X):
        X = self.conv_layers(X)
        X = self.fc_layers(X)
        return F.log_softmax(X, dim=1)

    @staticmethod
    def convBlock(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.nll_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.nll_loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.nll_loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)


# Define Architecture For CNN_TUMOR Model
class CNN_TUMOR(nn.Module):
    # Network Initialisation
    def __init__(self, params):
        super().__init__()
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # Convolution Layers
        
        h,w=findConv2dOutShape(Hin,Win,nn.Conv2d(Cin, init_f, kernel_size=3))
        h,w=findConv2dOutShape(h,w,nn.Conv2d(init_f,2*init_f, kernel_size=3))
        h,w=findConv2dOutShape(h,w,nn.Conv2d(2*init_f,4*init_f, kernel_size=3))
        h,w=findConv2dOutShape(h,w,nn.Conv2d(4*init_f, 8*init_f, kernel_size=3))
        # compute the flatten size
        self.num_flatten=h*w*8*init_f

        self.model = nn.Sequential(
            CNN_TUMOR.convBlock(Cin,init_f),
            CNN_TUMOR.convBlock(init_f,2*init_f),
            CNN_TUMOR.convBlock(2*init_f,4*init_f),
            CNN_TUMOR.convBlock(4*init_f,8*init_f),
            nn.Flatten(),
            nn.Linear(self.num_flatten, num_fc1),
            nn.Linear(num_fc1, num_classes)
            )

    def forward(self,X):
        return F.log_softmax(self.model(X), dim=1)
    
    # Define the Convolutional Block
    @staticmethod
    def convBlock(ni,no):
        return nn.Sequential(
            nn.Conv2d(ni, no, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


#  Class Activation Mapping (CAM) 
def im2gradCAM(model, image, verbose = False):
    model.eval()
    image_tensor = transform_single_image(image).to(torch.device('cpu'))
    im2fmap = nn.Sequential(*(list(model.model[:3])+list(model.model[3][:1])))
    logits = model(image_tensor)
    heatmap = []
    pred = logits.max(-1)[-1]
    model.zero_grad()
    logits[0,pred].backward(retain_graph=True)

    activations = im2fmap(image_tensor)
    pooled_grads = model.model[3][0].weight.grad.data.mean((1,2,3))
    if verbose:
        print(f"model.conv4.weight.grad.shape = {model.model[3][0].weight.grad.shape}")
        print(f"pooled_grads.shape = {pooled_grads.shape}")
        print(f"activations.shape={activations.shape}") 
    for i in range(activations.shape[1]):
        activations[:,i,:,:] *= pooled_grads[i] 
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    # Apply ReLU using the nn.ReLU() module
    relu = nn.ReLU()
    heatmap = relu(heatmap)
    overlaid_heatmap = upsampleHeatmap(heatmap,image)
    return overlaid_heatmap

def transform_single_image(image, image_size=(256,256)):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ]
    )  
    # Apply the transform to the image
    image_tensor = transform(image)

    # Add a batch dimension, since models usually expect batched inputs
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def upsampleHeatmap(heatmap, image):
    heatmap_ratio = 0.5 # This ratio is used to weight the contribution of the heatmap in the final overlaid image
    w,h = image.size
    m,M = heatmap.min(), heatmap.max()
    heatmap = 255 * ((heatmap-m) / (M-m))
    heatmap = np.uint8(heatmap)
    heatmap = cv2.resize(heatmap, (w,h))
    heatmap = cv2.applyColorMap(255-heatmap, cv2.COLORMAP_JET)
    heatmap = np.uint8(heatmap)
    heatmap = np.uint8(heatmap*heatmap_ratio + np.transpose(image,(0,1,2))*(1-heatmap_ratio))
    return heatmap


default_params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 4}


