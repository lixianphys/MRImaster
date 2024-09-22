import torch
from PIL import Image
from fastapi import (FastAPI, UploadFile, File, Request, HTTPException, Form)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.network import CNN_TUMOR, im2gradCAM
from src.utils import preprocess_image, CLA_label
import uuid
import os

best_model_wts = 'model_to_deploy.pt'

params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 4}

# 1. Load the model
model = CNN_TUMOR(params_model)  # Replace with your model class
model.load_state_dict(torch.load(best_model_wts,map_location=torch.device('cpu'),weights_only=True))  # Load the saved model weights
model.eval()  # Set model to evaluation mode

# 2. Predict function
def predict(image):
    # Preprocess the image
    image = preprocess_image(image)
    
    # Run the model on the input image
    with torch.no_grad():  # Disable gradient computation for faster inference
        output = model(image)
    
    # Optionally, apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Get the predicted class (assuming single-label classification)
    _, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), probabilities


app = FastAPI()

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files like CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize template engine
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/result", response_class=HTMLResponse)
async def result(
    request: Request, 
    file: UploadFile = File(None),
    original_image_url: str = Form(None),
    predicted_class = Form(None),
    probabilities = Form(None)):

    # Process the uploaded image

    if file:
        # Save the uploaded image
        img_filename = f"{uuid.uuid4().hex}.png"
        img_path = os.path.join(UPLOAD_DIR, img_filename)
        image = Image.open(file.file).convert("RGB")
        image.save(img_path)
        original_image_url = f"/static/uploads/{img_filename}"
        predicted_class, probabilities = predict(image)
        probabilities = f"{probabilities.tolist()[0][predicted_class]:0.3f}"
        predicted_class = CLA_label[predicted_class]
    elif not original_image_url:
        # Raise an error if neither file nor original_image_url is provided
        return HTMLResponse("Error: No image provided.", status_code=422)
    # Return the result page with the uploaded image URL
    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "original_image_url":  original_image_url
    })

@app.post("/apply-gradcam", response_class=HTMLResponse)
async def apply_operation(
    request: Request, 
    image_url: str = Form(...),
    predicted_class = Form(...),
    probabilities = Form(...)):
    # Open the image based on the provided URL
    img_path = os.path.join(UPLOAD_DIR, os.path.basename(image_url))
    image = Image.open(img_path)

    # Apply gradCAM
    new_image = Image.fromarray(im2gradCAM(model, image)) 

    # Save the new image
    new_img_filename = f"{uuid.uuid4().hex}.png"
    new_img_path = os.path.join(UPLOAD_DIR, new_img_filename)
    new_image.save(new_img_path)

    # Redirect to the operation result page
    return templates.TemplateResponse("apply_gradcam.html", {
        "request": request,
        "original_image_url": f"{image_url}",
        "new_image_url": f"/static/uploads/{new_img_filename}",
        "predicted_class": predicted_class,
        "probabilities": probabilities,
    })

# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


