import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from src.data_propessing.data_pipeline_nii import LazyLoadingNiftiDataset
from torch.utils.data import DataLoader
from src.unet3d import UNet3D


path = "../model_3dcnn_best.pt"
model = UNet3D(in_channels=4, out_channels=4)
model.load_state_dict(torch.load(path, weights_only=True))


# Paths to NIfTI images and labels
image_paths = ['../data/BRATS_484_img.nii','../data/BRATS_483_img.nii']
label_paths = ['../data/BRATS_484_lbl.nii','../data/BRATS_483_lbl.nii']
cache_dir = "../data/cache_dir"
# Initialize the dataset with caching
dataset = LazyLoadingNiftiDataset(image_paths=image_paths, label_paths=label_paths, cache_dir=cache_dir)

# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


# Switch model to evaluation mode
model.eval()
with torch.no_grad():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # Example for displaying a single batch (batch_size = 1)
    for inputs, labels in dataloader:
        inputs = inputs.to(device)  # Move inputs to the appropriate device
        labels = labels.to(device)  # Ground truth labels
        outputs = model(inputs)  # Run the model to get predictions
        outputs_soft = F.softmax(outputs, dim=1)
        # Convert model output to predicted class labels
        pred_labels = torch.argmax(outputs_soft, dim=1)  # Shape: (batch_size, H, W, D)

        # Move tensors to CPU for visualization
        inputs = inputs.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        labels = labels.cpu().numpy()


        # Display a few slices from the middle of each volume (e.g., middle slice in the depth dimension)
        slice_index = inputs.shape[2] // 3  # Choose a middle slice index in the depth dimension

        # Visualize side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Input - choose the first channel (e.g., FLAIR) or use a composite of multiple channels
        axes[0].imshow(inputs[0, 0, slice_index, :, :], cmap="gray")
        axes[0].set_title("Input (FLAIR)")

        # Prediction
        axes[1].imshow(pred_labels[0, slice_index, :, :], cmap="tab10")  # Use a colormap for segmentation
        axes[1].set_title("Predicted Segmentation")

        # Ground Truth
        axes[2].imshow(labels[0, slice_index, :, :], cmap="tab10")  # Use the same colormap for consistency
        axes[2].set_title("Ground Truth Segmentation")

        # Display the figure
        plt.show()
        