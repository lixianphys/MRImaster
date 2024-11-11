import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from src.data_propessing.data_pipeline_nii import LazyLoadingNiftiDataset
from torch.utils.data import DataLoader
from src.unet3d import UNet3D

def dice_loss(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(DiceCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        # Cross-entropy loss
        ce_loss = self.cross_entropy(outputs, labels)
        
        # Dice loss for each class
        outputs_soft = F.softmax(outputs, dim=1)  # Softmax along the channel dimension
        labels_one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        dice_loss_value = 0
        for i in range(outputs.shape[1]):  # For each class
            dice_loss_value += dice_loss(outputs_soft[:, i], labels_one_hot[:, i])
        
        # Combine the losses with equal weighting (adjust as needed)
        return ce_loss + dice_loss_value / outputs.shape[1]

def train(model,
          num_epochs,
          criterion,
          optimizer
          ):
    # Example training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for batch, (inputs, labels) in enumerate(dataloader):  # inputs shape: (batch_size, 4, H, W, D), labels shape: (batch_size, H, W, D)
            optimizer.zero_grad()
            
            outputs = model(inputs)  # outputs shape: (batch_size, num_classes, H, W, D)
            loss = criterion(outputs, labels)  # Compute the combined Dice + Cross-Entropy loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f"Batch {batch+1}/{len(dataloader)}, Loss: {loss.item()}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":

    # Paths to NIfTI images and labels
    image_paths = ['../data/BRATS_484_img.nii','../data/BRATS_483_img.nii']
    label_paths = ['../data/BRATS_484_lbl.nii','../data/BRATS_483_lbl.nii']
    cache_dir = "../data/cache_dir"
    # Initialize the dataset with caching
    dataset = LazyLoadingNiftiDataset(image_paths=image_paths, label_paths=label_paths, cache_dir=cache_dir)

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Instantiate the model and loss function
    model = UNet3D(in_channels=4, out_channels=4)  # 4 input channels, 4 output classes (background, edema, etc.)
    criterion = DiceCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Example training loop
    num_epochs = 50
    train(model,num_epochs,criterion,optimizer)
    path = "../model_3dcnn.pt"
    torch.save(model.state_dict(), path)
   
