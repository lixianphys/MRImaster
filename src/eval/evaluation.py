import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
import torch
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from src.utils.utils import (True_and_Pred, CLA_label,show_confusion_matrix, dice_score_per_class, iou_per_class)
from src.unet3d import UNet3D
from src.cnn import CNN_TUMOR
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
from src.preprocess.nifti import LazyLoadingNiftiDataset
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np


def eval_cnn(config):
    "Evaluate a CNN model"
    batch_size= config['evaluation']['batch_size']
    data_path = config['evaluation']['path']
    image_size = tuple(config['loading']['image_size'])
    print(f"Working on dataset at {data_path}")
    device=torch.device(config['evaluation']['device'])
    # define transformation
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ]
    )
    val_set = torchvision.datasets.ImageFolder(data_path,transform=transform)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 2)
    report_path = config['evaluation']['report_path']

    model = CNN_TUMOR(
    {
    'shape_in': tuple(config['model']['shape_in']),
    'num_classes':config['model']['num_classes'],
    'initial_filters': config['model']['initial_filters'],
    'num_fc1':config['model']['num_fc1'],
    'dropout_rate':config['model']['dropout_rate']
    }
    )

    model.load_state_dict(torch.load(config['evaluation']['model_wts'],weights_only=True))

    # check confusion matrix for error analysis
    model.eval()
    with torch.no_grad():
        y_true, y_pred = True_and_Pred(val_loader, model, device)

    report = classification_report(y_true, y_pred, output_dict=True)
    # Convert to DataFrame for better formatting
    report_df = pd.DataFrame(report).transpose()
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=CLA_label, columns=CLA_label)
    show_confusion_matrix(cm, CLA_label, title='Confusion matrix', cmap=plt.cm.YlGnBu)

    # Save the classification report as markdown
    with open(report_path, "w") as md_file:
        md_file.write("# Model Evaluation Report\n\n")
        md_file.write("## Classification Report\n\n")
        md_file.write(report_df.to_markdown() + "\n\n")
        md_file.write("## Confusion Matrix\n\n")
        md_file.write(cm_df.to_markdown() + "\n\n")


def eval_unet(config):
    " Evaluate a Unet3D model"
    in_channels=config['model']['in_channels']
    out_channels=config['model']['out_channels']
    num_classes = out_channels
    batch_size= config['evaluation']['batch_size']
    path = config['evaluation']['path']
    img_folder = os.path.join(path,'imageTr')
    # place all labels (nii or nii.gz) in data_path/labelTr
    lbl_folder = os.path.join(path,'labelTr')
    img_filenames = os.listdir(img_folder)
    lbl_filenames = os.listdir(lbl_folder)
    image_paths = [os.path.join(img_folder,file_path) for file_path in img_filenames]
    label_paths = [os.path.join(lbl_folder,file_path) for file_path in lbl_filenames]
    cache_dir = config['data']['cache_path']
    dataset = LazyLoadingNiftiDataset(image_paths=image_paths, label_paths=label_paths, cache_dir=cache_dir)
    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device  = torch.device(config['evaluation']['device'])
    model = UNet3D(in_channels,out_channels)
    model.load_state_dict(torch.load(config['evaluation']['model_wts'],weights_only=True))
    model.to(device)

    all_y_true = []
    all_y_pred = []
    dice_scores_all = []
    iou_scores_all = []   
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            # Get predictions
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)  # Convert logits to class indices
            
            # Compute Dice scores
            dice_scores = dice_score_per_class(predictions, labels, num_classes=num_classes)
            iou_scores = iou_per_class(predictions, labels, num_classes=num_classes)
            dice_scores_all.append(dice_scores)
            iou_scores_all.append(iou_scores)

            # Flatten for classification metrics
            all_y_true.extend(labels.cpu().numpy().flatten())
            all_y_pred.extend(predictions.cpu().numpy().flatten())

    # Average Dice scores per class
    avg_dice_scores = np.mean(dice_scores_all, axis=0)
    avg_iou_scores = np.mean(iou_scores_all, axis=0)

    # Overall metrics
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    class_report = classification_report(all_y_true, all_y_pred, output_dict=True)

    output_report_path = config['evaluation']['report_path']
    # Save report as Markdown
    with open(output_report_path, "w") as report_file:
        report_file.write("# Segmentation Evaluation Report\n\n")
        report_file.write("## Dice Scores Per Class\n")
        for cls, dice_score in enumerate(avg_dice_scores):
            report_file.write(f"- Class {cls}: Dice Score = {dice_score:.4f}\n")
        report_file.write("\n## IoU Scores Per Class\n")
        for cls, iou_score in enumerate(avg_iou_scores):
            report_file.write(f"- Class {cls}: IoU Score = {iou_score:.4f}\n")
        report_file.write("\n## Overall Accuracy\n")
        report_file.write(f"- Overall Accuracy: {overall_accuracy:.4f}\n\n")
        report_file.write("## Classification Metrics\n")
        for cls, metrics in class_report.items():
            if isinstance(metrics, dict):
                report_file.write(f"### Class {cls}\n")
                report_file.write(f"- Precision: {metrics['precision']:.4f}\n")
                report_file.write(f"- Recall: {metrics['recall']:.4f}\n")
                report_file.write(f"- F1-Score: {metrics['f1-score']:.4f}\n\n")

    print(f"Report saved to {output_report_path}")





