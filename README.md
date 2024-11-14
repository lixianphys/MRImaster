## MRIMaster: AI-supported medical imaging classifier and segmenter
### Table of Contents

[How to run the app](#How-to-run-the-app)
[Take a look at the app](#Take-a-look-at-the-app)
[Datasets for training](#Datasets-for-training)
[Data Preprocessing](#Data-Preprocessing)
[Model](#Model)
[Train](#Train)
[Deploy](#Deploy)
[Features to add](#features-to-add)

## For Users

### How to run the app

![Github](https://img.shields.io/badge/github-000000?logo=github)
```cmd
git clone git@github.com:lixianphys/MRImaster.git
cd mrimaster
git checkout published
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Take a look at the app
#### cnn model
| ![Image 1](frontend/static/app-description/2d-image-upload.png) | ![Image 2](frontend/static/app-description/2d-image-result.png) |
|-------------------------|------------------------|
#### unet3d model
| ![Image 1](frontend/static/app-description/3d-volume-upload.png) | ![Image 2](frontend/static/app-description/3d-volume-result.png) |
|-------------------------|------------------------|


## For developers
### Datasets for training 
#### Brats dataset - Task01 Brain Tumor (cnn model)
Brats2017 (Gliomas segmentation tumour and oedema in on brain images). "https://www.med.upenn.edu/sbia/brats2017.html"
This 4D image dataset contains brain MR images together with segmentation masks. All images and masks are provided in `.nii.gz` format with 4 channels (FLAIR,T1w, t1gd and T2w) per image. Masks are categorical with four classes: background, edema, non-enhancing tumor and enhancing tumour.

#### Kaggle dataset - brain-tumor-classification-mri (unet3d model)
This dataset contain Training and Testing folders. Each folder has four subfolders, which contain MRIs of respective tumor classes (Glioma, Meningioma, Pituitary and No Tumor) "https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri"

### Data Preprocessing
It is rather straightforward to download medium-sized, well-structured Kaggle dataset by using `src.preprocess.kaggledata.KaggleDataPipe`. While dealing with a large volume of `nii.gz` or `nii` files (a single file can exceed 100 Mb), we need to worry about how to reduce the loading time during training. For this consideration, please have a look at the design of `src.preprocess.nifti.LazyLoadingNiftiDataset` about caching and reloading.

### Model
For adapting models to more specific uses, some model hyperparameters, such as number of classes, can be modified directly at the `model` block in config files `config/cnn_cofig.yaml` and `config/unet_config.yaml`. Below are the default models for each type:
- **cnn model**: 4 layers of convoluational neural network for classification task. Input is in shape of (C=3, H=256, W=256). Output is the prediction of 4 classes.
- **unet3d model**: Unet shape for segmentation task. Input is in shape of (bach_size, C=4, H=128, W=128, D=128).

### Train
```
python scripts/train_model.py --model [cnn or unet3d] --config [path_to_config_file] --data_path [path_to_training_data, it will overwrite the config file] --use_mlflow
```
This command-line together with the config files for training different models (`cnn_config.yaml` and `unet_config.ymal`) provides a easy-to-go and flexible access to training your model. 

Before running this command-line, not only the most relevant `training` block in the config files should be checked and modified accordingly, but also the `data` and `mlflow` blocks should also be carefully scanned.

Additionally, adding `--use_mlflow` will log the experiment, parameters, metrics and artifacts into a MLflow server. Be sure that you have already spinned up one like this: 
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### Inference
Likewise, convienient inference directly from command-line is also offered:
```
python scripts/pred_model.py --model_type [cnn or unet3d] --config [path_to_config_file]
```
Modify the `config/inferecen_config.yaml` file, including the `model_path` pointing to the inference model. Be aware of the following hyperparameters (shape_in, num_classes, etc.), they should be compatible with the inference model. Last but not least, keep the value of `input_image_path` and `input_volume_path` updated.

For the cnn model, the prediction result is directly displayed. While the unet3d model would output a mask of predicted labels to the `output_path`, which should be modified accordingly.

### Deploy
In the previous single-modal version (`app_v0.py`), we used Fastapi framework to deploy this inference model locally. Here we adopt the Streamlit to deploy this multi-modal inference model (`app.py`). For more details about this app. Jump [here](#for-users) 

### Disclaimer
This dataset contains medical images intended solely for research, educational, and informational purposes.

### Features to add
- [x] Enable switching between models for different classification tasks
- [x] Build data pipeline for additional datasets beyond Kaggle, e.g., [TCIA API](https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guides)
- [x] Add object detection for identifying and measuring tumor size