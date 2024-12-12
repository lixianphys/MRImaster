import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Union


class LiUNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, dims=3, learning_rate=1e-3):
        super().__init__()
        
        # Save hyperparameters for flexibility
        self.save_hyperparameters()

        # Select appropriate convolution and pooling operations based on dimensions
        self.conv = getattr(nn, f'Conv{dims}d')
        self.conv_transpose = getattr(nn, f'ConvTranspose{dims}d')
        self.batch_norm = getattr(nn, f'BatchNorm{dims}d')
        self.max_pool = getattr(F, f'max_pool{dims}d')

        self.criterion = DiceCrossEntropyLoss()

        # Encoding path (down-sampling)
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoding path (up-sampling)
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Output layer
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.max_pool(enc1, 2))
        enc3 = self.enc3(self.max_pool(enc2, 2))
        enc4 = self.enc4(self.max_pool(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.max_pool(enc4, 2))
        
        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Output layer
        out = self.out_conv(dec1)
        return out
    
    def _conv_block(self,in_channels, out_channels):
        return nn.Sequential(
            self.conv(in_channels, out_channels, kernel_size=3, padding=1),
            self.batch_norm(out_channels),
            nn.ReLU(inplace=True),
            self.conv(out_channels, out_channels, kernel_size=3, padding=1),
            self.batch_norm(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {
            'loss': loss,
            'predictions': torch.argmax(y_hat, dim=1),
            'targets': y,
            'num_classes': self.hparams.out_channels
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log the loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {
            'loss': loss,
            'predictions': torch.argmax(y_hat, dim=1),
            'targets': y,
            'num_classes': self.hparams.out_channels
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # Encoding path (down-sampling)
        self.enc1 = UNet3D._conv_block(in_channels, 32)
        self.enc2 = UNet3D._conv_block(32, 64)
        self.enc3 = UNet3D._conv_block(64, 128)
        self.enc4 = UNet3D._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = UNet3D._conv_block(256, 512)
        
        # Decoding path (up-sampling)
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNet3D._conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNet3D._conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNet3D._conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNet3D._conv_block(64, 32)
        
        # Output layer
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        enc4 = self.enc4(F.max_pool3d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, 2))
        
        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Output layer
        out = self.out_conv(dec1)
        return out
    
    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, celoss_ratio:Union[float,int] =1, dims:int=3):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.celoss_ratio = celoss_ratio
        self.dims = dims

    def forward(self, outputs, labels):
        # Cross-entropy loss
        ce_loss = self.cross_entropy(outputs, labels)
        
        # Dice loss for each class
        outputs_soft = F.softmax(outputs, dim=1)  # Softmax along the class dimension
        if self.dims == 3:
            # For 3D: (B, C, D, H, W) -> (B, H, W, D, C)
            labels_one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).permute(0, 4, 1, 2, 3).float()
        else:  # 2D
            # For 2D: (B, C, H, W) -> (B, H, W, C)
            labels_one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
        
        dice_loss_value = self.dice_loss(outputs_soft, labels_one_hot)
        
       # Combine losses
        total_loss = ce_loss*self.celoss_ratio + dice_loss_value
        return total_loss
    
    def dice_loss(self,outputs_soft, labels_one_hot, smooth=1.0):
        """
        Compute the average Dice loss across all classes.
        """
        dice_loss_per_class = []
        for c in range(outputs_soft.shape[1]):  # Loop over each class
            pred = outputs_soft[:, c]  # Softmax probability for class `c`
            target = labels_one_hot[:, c]  # One-hot encoded ground truth for class `c`
            dims = tuple(range(1, self.dims+1))
            intersection = (pred * target).sum(dim=dims)  # Sum over spatial dimensions
            dice = (2.0 * intersection + smooth) / (pred.sum(dim=dims) + target.sum(dim=dims) + smooth)
            dice_loss_per_class.append(1 - dice)  # Dice loss for class `i`
        # Average Dice loss over batch and classes
        dice_loss = torch.mean(torch.stack(dice_loss_per_class, dim=1), dim=1)  # Average over classes
        return dice_loss.mean()  # Average over batch
