import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

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

# Define the model class (this should match the class structure when the model was saved)
class CNN_TUMOR(nn.Module):
    
    # Network Initialisation
    def __init__(self, params):
        
        super(CNN_TUMOR, self).__init__()
    
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2}
# 1. Load the model
model = CNN_TUMOR(params_model)  # Replace with your model class
model.load_state_dict(torch.load('data/weights.pt',map_location=torch.device('cpu'),weights_only=True))  # Load the saved model weights
model.eval()  # Set model to evaluation mode

# 2. Image Preprocessing (resize, normalize, convert to tensor)
def preprocess_image(image):
    # Define the transformations (modify based on your model's requirements)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to 224x224 (adjust as per your model)
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])
    
    # Apply the transformations
    image = transform(image)
    
    # Add batch dimension (1, C, H, W) because model expects a batch of images
    image = image.unsqueeze(0)
    
    return image

# 3. Predict function
def predict(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Run the model on the input image
    with torch.no_grad():  # Disable gradient computation for faster inference
        output = model(image)
    
    # Optionally, apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Get the predicted class (assuming single-label classification)
    _, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), probabilities

CLA_label = {
    0 : 'Brain Tumor',
    1 : 'Healthy'
} 


templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the prediction route
@app.post("/result", response_class=HTMLResponse)
async def result(request: Request, file: UploadFile = File(...)):
    # Process the uploaded image
    image = Image.open(file.file).convert('RGB')
    predicted_class, probabilities = predict(image)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": CLA_label[predicted_class],
        "probabilities": probabilities.tolist()[0][predicted_class]
    })


# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


