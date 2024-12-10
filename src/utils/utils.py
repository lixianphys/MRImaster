import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import shutil
import os


# Label Mapping
CLA_label = {
    0 : "glioma tumor",
    1 : "meningioma tumor",
    2 : "no tumor",
    3 : "pituitary tumor"
}

def check_and_prepare_file_path(file_path:str) -> str:
    """
    Check if the directory for a given .pt file exists.
    If the directory doesn't exist, create it.

    Args:
        file_path (str): The full path to the .pt file.
    """
    # Convert the file path to a Path object
    file_path = Path(file_path)
    
    # Extract the directory
    directory = file_path.parent
    
    # Check if the directory exists
    if not directory.exists():
        # Create the directory
        directory.mkdir(parents=True, exist_ok=True)
        
    return file_path

def delete_folder_with_confirmation(folder_path: str) -> None:
    """
    Deletes a folder and all its contents with user confirmation.

    Args:
        folder_path (str): Path to the folder to delete.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Ask for user confirmation
    confirmation = input(f"Are you sure you want to permanently delete the folder '{folder_path}' and all its contents? (yes/no): ").strip().lower()
    
    if confirmation == "yes":
        try:
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and its contents have been deleted.")
        except PermissionError:
            print(f"Permission denied: Unable to delete '{folder_path}'.")
        except Exception as e:
            print(f"An error occurred while deleting the folder: {e}")
    else:
        print("Deletion cancelled.")

def delete_folder_without_confirmation(folder_path: str) -> None:
    """
    Deletes a folder and all its contents with user confirmation.

    Args:
        folder_path (str): Path to the folder to delete.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    except PermissionError:
        print(f"Permission denied: Unable to delete '{folder_path}'.")
    except Exception as e:
        print(f"An error occurred while deleting the folder: {e}")




def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any directory.

    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)

def timeit(fn): 
    import time
    
    # *args and **kwargs are to support positional and named arguments of fn
    def get_time(*args, **kwargs): 
        start = time.time() 
        output = fn(*args, **kwargs)
        print(f"Time taken in {fn.__name__}: {time.time() - start:.3f} seconds.")
        return output  # make sure that the decorator returns the output of fn
    return get_time

# Image Preprocessing (resize, normalize, convert to tensor)
def preprocess_image(image):
    # Define the transformations (modify based on your model's requirements)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to 256x256 (adjust as per your model)
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])
    
    # Apply the transformations
    image = transform(image)
    
    # Add batch dimension (1, C, H, W) because model expects a batch of images
    image = image.unsqueeze(0)
    
    return image


# Function to get the learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# Function to compute the loss value per batch of data
def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target) # get loss
    pred = output.argmax(dim=1, keepdim=True) # Get Output Class
    metric_b=pred.eq(target.view_as(pred)).sum().item() # get performance metric
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# Compute the loss value & performance metric for the entire dataset (epoch)
def loss_epoch(model,device,loss_func,dataset_dl,opt=None):
    
    run_loss=0.0 
    t_metric=0.0
    len_data=len(dataset_dl.dataset)
    model = model.to(device)
    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb) # get model output
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt) # get loss per batch
        run_loss+=loss_b        # update running loss

        if metric_b is not None: # update running metric
            t_metric+=metric_b    
    
    loss=run_loss/float(len_data)  # average loss value
    metric=t_metric/float(len_data) # average metric value
    
    return loss, metric


# define function For Classification Report
def True_and_Pred(val_loader, model, device):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        model = model.to(device)
        labels = labels.numpy()
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        pred = pred.detach().cpu().numpy()
        
        y_true = np.append(y_true, labels)
        y_pred = np.append(y_pred, pred)
    
    return y_true, y_pred


# Confusion Matrix Plotting Function
def show_confusion_matrix(cm, CLA_label, title='Confusion matrix', cmap=plt.cm.YlGnBu):
    
    plt.figure(figsize=(10,7))
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLA_label))

    plt.xticks(tick_marks, [f"{value}={key}" for key , value in CLA_label.items()], rotation=45)
    plt.yticks(tick_marks, [f"{value}={key}" for key , value in CLA_label.items()])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i,j]}\n{cm[i,j]/np.sum(cm)*100:.2f}%", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


def dice_score_per_class(pred, target, num_classes, smooth=1.0):
    """
    Compute the Dice score for each class.
    Args:
        pred: Predicted class tensor (batch, height, width, depth).
        target: Ground truth class tensor (batch, height, width, depth).
        num_classes: Total number of classes.
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        List of Dice scores per class.
    """
    dice_scores = []
    for cls in range(num_classes):
        pred_bin = (pred == cls).float()
        target_bin = (target == cls).float()
        intersection = (pred_bin * target_bin).sum()
        union = pred_bin.sum() + target_bin.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    return dice_scores

def iou_per_class(pred, target, num_classes, smooth=1.0):
    """
    Compute IoU (Intersection over Union) for each class.
    Args:
        pred: Predicted class tensor (batch, height, width, depth).
        target: Ground truth class tensor (batch, height, width, depth).
        num_classes: Total number of classes.
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        List of IoU scores per class.
    """
    iou_scores = []
    for cls in range(num_classes):
        pred_bin = (pred == cls).float()
        target_bin = (target == cls).float()
        intersection = (pred_bin * target_bin).sum()
        union = pred_bin.sum() + target_bin.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    return iou_scores

def load_config_from_yaml(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    print(f"run your quick test here for all functions/classes in this script {__file__}.")
    print("-----------------------------------------------------------")
    count_calls = Counter()
    for i in range(5):
        print(count_calls())
