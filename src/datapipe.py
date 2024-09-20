import kaggle as kg
import boto3
from dotenv import load_dotenv
from pathlib import Path
import os


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
