import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
from src.preprocess.kaggledata import KaggleDataPipe


if __name__ == "__main__":

    kaggle_link = "sartajbhuvaji/brain-tumor-classification-mri"
    dir_to_store = "data/raw_data/brain-tumor-classification-mri"
    cnn_dataset = KaggleDataPipe(kaggle_link,dir_to_store)
    cnn_dataset.load_from_kaggle()
