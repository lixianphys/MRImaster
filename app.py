import streamlit as st
import os
import torch
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from src.inference.predict import (
    load_cnn_model, load_unet3d_model,
    preprocess_image, preprocess_volume,
    cnn_inference, unet3d_inference
)
from src.utils.configYaml import load_config_from_yaml, string_tuple_to_tuple
from src.cnn import im2gradCAM
from src.utils.utils import CLA_label
import matplotlib.pyplot as plt
import tempfile


cnn_config = load_config_from_yaml("config/cnn.yaml")
unet_config = load_config_from_yaml("config/unet.yaml")

# Define modalities and labels
modalities = {
    "0": "FLAIR",
    "1": "T1w",
    "2": "t1gd",
    "3": "T2w"
}

labels = {
    "0": "background",
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour"
}

axes = {
    "0": "Sagittal",
    "1": "Coronal",
    "2": "Axial"
}

colors = {
    "background": (0, 0, 0, 0),  # transparent
    "edema": (0.2, 0.8, 0.2),    # green
    "non-enhancing tumor": (1.0, 0.6, 0.2),  # orange
    "enhancing tumour": (0.8, 0.2, 0.2)  # red
}
# Separate cache functions for each model
@st.cache_resource
def load_cnn_model_cached():
    return load_cnn_model(
        model_path=cnn_config['deploy']['model'], 
        device=torch.device(cnn_config['deploy']['device']), 
        params={
            "shape_in": string_tuple_to_tuple(cnn_config['model']['shape_in']),
            "num_classes": cnn_config['model']['num_classes'],
            "initial_filters": cnn_config['model']['initial_filters'],
            "num_fc1": cnn_config['model']['num_fc1'],
            "dropout_rate": cnn_config['model']['dropout_rate']
        }
    )

@st.cache_resource
def load_unet3d_model_cached():
    return load_unet3d_model(
        model_path=unet_config['eval']['model'], 
        device=torch.device(unet_config['deploy']['device']), 
        in_channels=unet_config['model']['in_channels'], 
        out_channels=unet_config['model']['out_channels']
    )

def main():
    st.title("Medical Imaging Classifier and Segmenter")
    st.write("Select a model and upload an image or 3D volume for inference.")

    # Select the model type
    model_type = st.selectbox("Choose the Model", ["CNN (2D Image Classification)", "UNet3D (3D Volume Segmentation)"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load only the selected model
    if model_type == "CNN (2D Image Classification)":
        cnn_model = load_cnn_model_cached()
        unet3d_model = None
    else:
        unet3d_model = load_unet3d_model_cached()
        cnn_model = None
    
    if model_type == "CNN (2D Image Classification)":
        st.subheader("2D Image Classification with CNN")
        uploaded_image = st.file_uploader("Upload a 2D Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Load and display the image
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", width=200)
            
            # Add transparency slider
            overlay_alpha = st.slider("Grad-CAM Overlay Transparency", 0.0, 1.0, 0.5)
            
            # Get Grad-CAM image and convert to RGBA
            cam_image = Image.fromarray(im2gradCAM(cnn_model, image))
            cam_rgba = cam_image.convert("RGBA")
            
            # Apply transparency
            data = cam_rgba.getdata()
            new_data = [(r, g, b, int(255 * overlay_alpha)) for (r, g, b, a) in data]
            cam_rgba.putdata(new_data)
            
            st.image(cam_rgba, caption="Grad-CAM Image", width=200)

            # Preprocess and inference
            image_tensor = preprocess_image(uploaded_image, device)
            prediction = cnn_inference(cnn_model, image_tensor)
            st.write(f"**Prediction:** Class {CLA_label[prediction]}")

            # Optionally save Grad-CAM image
            save_output = st.checkbox("Save Grad-CAM as png file")
            if save_output:
                cam_rgba.save("output/output_grad_cam_image.png")
                st.write("**Output saved to `output/output_grad_cam_image.png`**")

    elif model_type == "UNet3D (3D Volume Segmentation)":
        st.subheader("3D Volume Segmentation with UNet3D")
        uploaded_volume = st.file_uploader("Upload a 3D Volume (NIfTI format)", type=["nii", "nii.gz"])
        
        # Add optional ground truth upload
        uploaded_label = st.file_uploader("(Optional) Upload Ground Truth Labels (NIfTI format)", type=["nii", "nii.gz"])
        
        if uploaded_volume is not None:
            @st.cache_data
            def process_volume(volume_data, is_label=False):
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                    tmp_file.write(volume_data.getvalue())
                    file_path = tmp_file.name
                
                try:
                    return preprocess_volume(file_path, device, is_label)
                finally:
                    os.unlink(file_path)

            @st.cache_data(show_spinner=False)
            def get_prediction(_volume_tensor):
                return unet3d_inference(unet3d_model, _volume_tensor)

            # Process volume and get prediction
            volume_tensor = process_volume(uploaded_volume)
            predicted_volume = get_prediction(volume_tensor)
            
            # Process ground truth if provided
            ground_truth = None
            if uploaded_label is not None:
                ground_truth = process_volume(uploaded_label, is_label=True)

            # Get the volume data in the correct shape
            volume_data = volume_tensor.cpu().numpy()  # Convert tensor to numpy array
            # UI controls
            col1, col2, col3 = st.columns(3)
            with col1:
                modality_choice = st.selectbox("MRI Modality", options=list(modalities.values()))
                modality_key = next((k for k, v in modalities.items() if v == modality_choice), None)
            with col2:
                axis_choice = st.selectbox("View Axis", options=list(axes.values()))
                axis_key = next((k for k, v in axes.items() if v == axis_choice), None)
            with col3:
                overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)

            # Get slice index based on the selected axis
            slice_idx = st.slider("Select Slice", 0, volume_data.shape[int(axis_key)+2] - 1, 
                                volume_data.shape[int(axis_key)+2] // 2)

            # Create the figure and axes
            fig, ax = plt.subplots(1, 2 if ground_truth is not None else 1, 
                                 figsize=(15 if ground_truth is not None else 10, 5))
            if ground_truth is not None:
                ax = np.atleast_1d(ax)
            else:
                ax = [ax]

            # Get MRI slice based on the selected axis and modality
            mri_slice = {
                "0": volume_data[0, int(modality_key), slice_idx, :, :],
                "1": volume_data[0, int(modality_key), :, slice_idx, :],
                "2": volume_data[0, int(modality_key), :, :, slice_idx]
            }[axis_key]

            pred_slice = {
                "0": predicted_volume[0, slice_idx, :, :],
                "1": predicted_volume[0, :, slice_idx, :],
                "2": predicted_volume[0, :, :, slice_idx]
            }[axis_key]

            # Plot prediction
            ax[0].imshow(mri_slice, cmap='gray')
            masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
            ax[0].imshow(masked_pred, cmap=ListedColormap([colors[label] for label in labels.values()]), 
                          alpha=overlay_alpha)
            ax[0].set_title("Prediction")

            # Plot ground truth if available
            if ground_truth is not None:
                gt_slice = {
                    "0": ground_truth[0,slice_idx, :, :],
                    "1": ground_truth[0, :, slice_idx, :],
                    "2": ground_truth[0, :, :, slice_idx]
                }[axis_key]
                
                # ax[1].imshow(mri_slice, cmap='gray')
                # masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
                ax[1].imshow(gt_slice, cmap=ListedColormap([colors[label] for label in labels.values()]), 
                              alpha=overlay_alpha)
                ax[1].set_title("Ground Truth")

            plt.tight_layout()
            st.pyplot(fig)

            # Legend
            st.subheader("Legend")
            cols = st.columns(len(labels))
            for idx, (_, label_name) in enumerate(labels.items()):
                with cols[idx]:
                    color = colors[label_name]
                    st.markdown(f'<span style="color:rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1);">⬤</span> {label_name}', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
