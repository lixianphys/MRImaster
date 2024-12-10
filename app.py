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

# Load models once and cache them
@st.cache_resource
def load_models():
    cnn_model = load_cnn_model(
        model_path=cnn_config['deploy']['model'], 
        device = torch.device(cnn_config['deploy']['device']), 
        params = {
        "shape_in":string_tuple_to_tuple(cnn_config['model']['shape_in']),
        "num_classes":cnn_config['model']['num_classes'],
        "initial_filters":cnn_config['model']['initial_filters'],
        "num_fc1":cnn_config['model']['num_fc1'],
        "dropout_rate":cnn_config['model']['dropout_rate']
        }
    )
    unet3d_model = load_unet3d_model(
        model_path=unet_config['eval']['model'], 
        device = torch.device(unet_config['deploy']['device']), 
        in_channels = unet_config['model']['in_channels'], 
        out_channels = unet_config['model']['out_channels']
    )
    return cnn_model, unet3d_model

def main():
    st.title("Medical Imaging Classifier and Segmenter")
    st.write("Select a model and upload an image or 3D volume for inference.")

    # Select the model type
    model_type = st.selectbox("Choose the Model", ["CNN (2D Image Classification)", "UNet3D (3D Volume Segmentation)"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    cnn_model, unet3d_model = load_models()
    
    if model_type == "CNN (2D Image Classification)":
        st.subheader("2D Image Classification with CNN")
        uploaded_image = st.file_uploader("Upload a 2D Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Load and display the image
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", width=200)
            cam_image = Image.fromarray(im2gradCAM(cnn_model, image)) 
            st.image(cam_image, caption="Grad-CAM Image",width=200)


            # Preprocess and inference
            image_tensor = preprocess_image(uploaded_image, device)
            prediction = cnn_inference(cnn_model, image_tensor)
            st.write(f"**Prediction:** Class {CLA_label[prediction]}")

            # Optionally save Grad-CAM image
            save_output = st.checkbox("Save Grad-CAM as png file")
            if save_output:
                cam_image.save("output/output_grad_cam_image.png")
                st.write("**Output saved to `output/output_grad_cam_image.png`**")

    elif model_type == "UNet3D (3D Volume Segmentation)":
        st.subheader("3D Volume Segmentation with UNet3D")
        uploaded_volume = st.file_uploader("Upload a 3D Volume (NIfTI format)", type=["nii", "nii.gz"])
        if uploaded_volume is not None:
            # Directly pass the uploaded volume to preprocess_volume
            volume_path = os.path.join("data/imageTr",uploaded_volume.name)

            volume_tensor = preprocess_volume(volume_path, device)
            predicted_volume = unet3d_inference(unet3d_model, volume_tensor)

            # Streamlit options for modality and colormap
            st.subheader("MRI Modality")

            # Choose modality
            modality_choice = st.selectbox("Choose MRI Modality", options=list(modalities.values()))
            st.write(f"Selected Modality: {modality_choice}")

            modality_key = next((k for k, v in modalities.items() if v == modality_choice), None)

            axis_choice = st.selectbox("Choose the axis to display", options=list(axes.values()))
            st.write(f"Selected Modality: {axis_choice}")

            axis_key = next((k for k, v in axes.items() if v == axis_choice), None)

            # Define color map based on labels
            colors = {
            "background": (0, 0, 0, 1),       # Black
            "edema": (0.6, 0.8, 0.2, 1),      # Light green
            "non-enhancing tumor": (0.3, 0.3, 0.7, 1),  # Blueish
            "enhancing tumour": (1, 0.5, 0, 1)  # Orange
            }
            cmap = ListedColormap([colors[label] for label in labels.values()])


            # Display the segmented result
            st.write("**Segmentation Result:**")
            st.write("Displaying one slice of the 3D segmentation")
            slice_idx = st.slider("Select Slice", 0, predicted_volume.shape[3] - 1, predicted_volume.shape[3] // 2)
            permuted_volume = volume_tensor.permute(0,2,3,4,1).to(device).numpy()


            volume_slice = {
                "0": permuted_volume[0, slice_idx, :, :,int(modality_key)],
                "1": permuted_volume[0, :, slice_idx, :,int(modality_key)],
                "2": permuted_volume[0, :, :, slice_idx,int(modality_key)]
            }[axis_key]
            st.image(volume_slice,caption =f"Input Slice {slice_idx}",width=200)

            predicted_slice = {
                "0": predicted_volume[0, slice_idx, :, :],
                "1": predicted_volume[0, :, slice_idx, :],
                "2": predicted_volume[0, :, :, slice_idx]
            }[axis_key]
            image_array = predicted_slice
            colored_image = cmap(image_array / 3)
            st.image((colored_image * 255).astype(np.uint8),caption =f"Segmentation Slice {slice_idx}",width=200)

            for _, label_name in labels.items():
                color_patch = f"rgba({int(colors[label_name][0]*255)}, {int(colors[label_name][1]*255)}, {int(colors[label_name][2]*255)}, 1)"
                st.markdown(f'<span style="color:{color_patch}; font-weight:bold;">⬤</span> {label_name}', unsafe_allow_html=True)

            # Optionally save output
            save_output = st.checkbox("Save output as NIfTI file")
            if save_output:
                predicted_volume_nifti = nib.Nifti1Image(predicted_volume.astype(np.float32), affine=np.eye(4))
                nib.save(predicted_volume_nifti, "output/segmented_volume.nii")
                st.write("**Output saved to `output/segmented_volume.nii`**")

if __name__ == "__main__":
    main()
