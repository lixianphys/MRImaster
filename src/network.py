import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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

# Define the Convolutional Block
def convBlock(ni,no):
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

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
            convBlock(Cin,init_f),
            convBlock(init_f,2*init_f),
            convBlock(2*init_f,4*init_f),
            convBlock(4*init_f,8*init_f),
            nn.Flatten(),
            nn.Linear(self.num_flatten, num_fc1),
            nn.Linear(num_fc1, num_classes)
            )

    def forward(self,X):
        return F.log_softmax(self.model(X), dim=1)
    

default_params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2}


