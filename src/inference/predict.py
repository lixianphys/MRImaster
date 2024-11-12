# predict_functions.py
import torch
import nibabel as nib
import numpy as np
from PIL import Image
from torchvision import transforms
from src.network import CNN_TUMOR # Import the CNN model architecture
from src.unet3d import UNet3D

def load_cnn_model(model_path, device, params):
    model = CNN_TUMOR(params).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True,map_location=device))
    model.eval()
    return model

def load_unet3d_model(model_path, device, in_channels=1, out_channels=1):
    model = UNet3D(in_channels=in_channels, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True,map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def preprocess_volume(volume_path, device):
    def center_crop(img, target_shape=(128,128,128)):
        """Crop the center of the image to the target shape."""
        crop_slices = tuple(
            slice((dim - target) // 2, (dim - target) // 2 + target)
            for dim, target in zip(img.shape, target_shape)
        )
        return img[crop_slices]
    volume = nib.load(volume_path).get_fdata()
    volume = center_crop(volume)
    volume = (volume-np.mean(volume))/np.std(volume)
    volume = np.clip(volume, 0, 1)
    volume = torch.tensor(volume, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
    return volume.to(device)

def cnn_inference(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()  # Return the class label
    
def unet3d_inference(model, volume_tensor):
    with torch.no_grad():
        outputs = model(volume_tensor)
        # Convert model output to predicted class labels
        predicted_volume = torch.argmax(outputs, dim=1)
        return predicted_volume.cpu().numpy()  # Convert to numpy for saving


def visualize_unet3d_prediction(inputs,pred_labels):

    import matplotlib.pyplot as plt

    slice_index = inputs.shape[2] // 3  # Choose a middle slice index in the depth dimension
    # Visualize side-by-side
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Input - choose the first channel (e.g., FLAIR) or use a composite of multiple channels
    axes[0].imshow(inputs[0, 0, slice_index, :, :], cmap="gray")
    axes[0].set_title("Input (FLAIR)")

    # Prediction
    axes[1].imshow(pred_labels[0, slice_index, :, :], cmap="tab10")  # Use a colormap for segmentation
    axes[1].set_title("Predicted Segmentation")


    # Display the figure
    plt.show()
        