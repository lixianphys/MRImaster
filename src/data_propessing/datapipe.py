import kaggle as kg
import boto3
from dotenv import load_dotenv
from pathlib import Path
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go



nii_data_info = {
    "name": "BRATS", 
    "description": "Gliomas segmentation tumour and oedema in on brain images",
    "reference": "https://www.med.upenn.edu/sbia/brats2017.html",
    "licence":"CC-BY-SA 4.0",
    "release":"2.0 04/05/2018",
    "tensorImageSize": "4D",
    "modality": { 
        "0": "FLAIR", 
        "1": "T1w", 
        "2": "t1gd",
        "3": "T2w"
    },  
    "labels": { 
        "0": "background", 
        "1": "edema",
        "2": "non-enhancing tumor",
        "3": "enhancing tumour"
    }
}


class KaggleDataPipe(object):
    def __init__(self,kaggle_link:str,dir_to_store:str) -> None:
        try:
            kg.api.authenticate()
            print("Authentication to Kaggle successful!")
        except Exception as e:
            print(f"Authentication failed! Error: {e}")
        self.link = kaggle_link
        self.dir_to_store = dir_to_store

    def load_from_kaggle(self) -> None:
        try:
            kg.api.dataset_download_files(dataset=self.link,
                                        path=self.dir_to_store,
                                        unzip=True)
            print(f"File download successful!")
        except Exception as e:
            print(f"Download failed! Error: {e}")

    def get_labels(self,folder_name=None):
        if folder_name is None:
            folder_path = Path(self.dir_to_store)            
            # List subdirectories in the specified folder
            subfolders = [f.name for f in folder_path.iterdir() if f.is_dir()]
            # Return the list of subfolders and the relative path
            return subfolders

        for root, dirs, files in os.walk(self.dir_to_store):
            if folder_name in dirs:
                folder_path = Path(root) / folder_name             
                # List subdirectories in the specified folder
                subfolders = [f.name for f in folder_path.iterdir() if f.is_dir()]
                
                # Return the list of subfolders and the relative path
                return subfolders
                
        raise FileNotFoundError(f"Folder '{folder_name}' not found in '{self.dir_to_store}'recursively.")


if __name__ == "__main__":
    # Load the NIfTI file
    file_path = '../data/BRATS_484_lbl.nii'
    nifti_img = nib.load(file_path)

    # Access the data as a NumPy array
    image_data = nifti_img.get_fdata()
    print("Image shape:", image_data.shape)

    # # Get affine matrix (used for spatial orientation)
    # affine_matrix = nifti_img.affine
    # print("Affine matrix:\n", affine_matrix)

    # # Header information
    # header = nifti_img.header
    # print("Header information:\n", header)

    print(np.unique(image_data))
    # Choose slices to display
    slice_x = image_data[image_data.shape[0] // 3, :, :]
    slice_y = image_data[:, image_data.shape[1] // 3, :]
    slice_z = image_data[:, :, image_data.shape[2] // 2]

    # Plot slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(slice_x.T, cmap="jet", origin="lower")
    axes[0].set_title("Sagittal Slice")
    axes[1].imshow(slice_y.T, cmap="jet", origin="lower")
    axes[1].set_title("Coronal Slice")
    axes[2].imshow(slice_z.T, cmap="jet", origin="lower")
    axes[2].set_title("Axial Slice")
    plt.show()

# load_dotenv(verbose=True, dotenv_path='.env')
# my_bucket = os.getenv('S3_BUCKET_ADDRESS')
# print(f"S3_BUCKET_ADDRESS: {my_bucket}")


# file_path = 'data/raw_data/'

# # Checking authentication
# try:
#     kg.api.authenticate()
#     print("Authentication to Kaggle successful!")
# except Exception as e:
#     print(f"Authentication failed! Error: {e}")


# # Attempting download
# try:
#     file_path = 'data/raw_data/'
#     kg.api.dataset_download_files(dataset="preetviradiya/brian-tumor-dataset",
#                                   path=file_path,
#                                   unzip=True)
#     print(f"File download successful! Data is in {file_path}")
# except Exception as e:
#     print(f"Download failed! Error: {e}")

# directory_path = Path('data/raw_data/')

# # Create an S3 client
# s3 = boto3.resource('s3')
# bucket = s3.Bucket(my_bucket)

# print('Attempting upload to S3 bucket:')

# try:
#     my_bucket = os.getenv('S3_BUCKET_ADDRESS')
#     if my_bucket is None:
#         raise ValueError("Environment variable 'S3_BUCKET_ADDRESS' not set")
# except Exception as e:
#     print(f"Error getting environment variable: {e}")

# # Recursively walk through the directory and subdirectories
# for file_path in directory_path.rglob('*'):  # rglob() recursively matches all files and subdirectories
#     if file_path.is_file():
#         # Generate the S3 key (remote path), keeping the folder structure
#         s3_key = file_path.relative_to(directory_path)

#         try:
#             # Upload file to S3 with the correct S3 key
#             bucket.upload_file(str(file_path), f'raw_data/{s3_key}')
#             print(f"Uploaded {s3_key}")
#         except Exception as e:
#             print(f"Error uploading {file_path}: {e}")

# print(f"Successfully uploaded raw data to {my_bucket}/raw_data")
