import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.network import CNN_TUMOR
from src.utils import preprocess_image, CLA_label

best_model_wts = 'weights.pt'

app = FastAPI()

params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2}

# 1. Load the model
model = CNN_TUMOR(params_model)  # Replace with your model class
model.load_state_dict(torch.load(best_model_wts,map_location=torch.device('cpu'),weights_only=True))  # Load the saved model weights
model.eval()  # Set model to evaluation mode

# 2. Predict function
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


