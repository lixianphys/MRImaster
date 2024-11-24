import kaggle as kg
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