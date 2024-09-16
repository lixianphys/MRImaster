import kaggle as kg
import boto3
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(verbose=True, dotenv_path='.env')
my_bucket = os.getenv('S3_BUCKET_ADDRESS')
print(f"S3_BUCKET_ADDRESS: {my_bucket}")


file_path = 'data/raw_data/'

# Checking authentication
try:
    kg.api.authenticate()
    print("Authentication to Kaggle successful!")
except Exception as e:
    print(f"Authentication failed! Error: {e}")


# Attempting download
try:
    file_path = 'data/raw_data/'
    kg.api.dataset_download_files(dataset="preetviradiya/brian-tumor-dataset",
                                  path=file_path,
                                  unzip=True)
    print(f"File download successful! Data is in {file_path}")
except Exception as e:
    print(f"Download failed! Error: {e}")

directory_path = Path('data/raw_data/')

# Create an S3 client
s3 = boto3.resource('s3')
bucket = s3.Bucket(my_bucket)

print('Attempting upload to S3 bucket:')

try:
    my_bucket = os.getenv('S3_BUCKET_ADDRESS')
    if my_bucket is None:
        raise ValueError("Environment variable 'S3_BUCKET_ADDRESS' not set")
except Exception as e:
    print(f"Error getting environment variable: {e}")

# Recursively walk through the directory and subdirectories
for file_path in directory_path.rglob('*'):  # rglob() recursively matches all files and subdirectories
    if file_path.is_file():
        # Generate the S3 key (remote path), keeping the folder structure
        s3_key = file_path.relative_to(directory_path)

        try:
            # Upload file to S3 with the correct S3 key
            bucket.upload_file(str(file_path), f'raw_data/{s3_key}')
            print(f"Uploaded {s3_key}")
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")

print(f"Successfully uploaded raw data to {my_bucket}/raw_data")
