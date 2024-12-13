import monai
from monai.data import ImageDataset, DataLoader
import os
from glob import glob
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, Compose, Resize, RandRotate90, RandSpatialCrop


class clfDataset(ImageDataset):
    """
    Dataset for 2d classification tasks. 
    This class provides a dataset for 2D medical image classification tasks.
    It inherits from MONAI's ImageDataset and provides convenient functionality
    for loading and transforming 2D medical images.

    Example usage:
        # Create training dataset with default transforms
        train_ds = clfDataset.from_default(
            image_dir='path/to/train/images',
            is_train=True,
            size=(256, 256)
        )

        # Create validation dataset
        val_ds = clfDataset.from_default(
            image_dir='path/to/val/images', 
            is_train=False,
            size=(256, 256)
        )

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

    Args:
        image_dir (str): Directory containing the image files
        transform (Transform, optional): Transform to be applied to the images.
            If None, no transform is applied.
    """
    def __init__(self,
                 image_dir,
                 transform=None):
        super().__init__(image_files=glob(os.path.join(image_dir,"**")),
                         transform=transform)
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self,index):
        return super().__getitem__(index)
    
    @classmethod
    def from_default(cls,image_dir,is_train=True,size=(256,256)):
        if is_train:
            clf2d_train_transform = Compose([LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize(spatial_size=size),
            RandRotate90(prob=0.5)])
            return cls(image_dir,transform=clf2d_train_transform)
        else:  
            clf2d_val_transform = Compose([LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize(spatial_size=size)])
            return cls(image_dir,transform=clf2d_val_transform)


class segDataset(ImageDataset):
    """
    Dataset for 3D segmentation tasks.
    This class provides a dataset for 3D medical image segmentation tasks.
    It inherits from MONAI's ImageDataset and provides convenient functionality
    for loading and transforming 3D medical images.

    Example usage:
        # Create training dataset with default transforms
        train_ds = segDataset.from_default(
            image_dir='data/imageTr',  # Directory containing 3D image volumes (e.g. NIfTI files)
            seg_dir='data/labelTr',    # Directory containing segmentation masks
            is_train=True,
            chunck_size=(64, 64, 64)   # Size of random crops during training
        )

        # Create validation dataset
        val_ds = segDataset.from_default(
            image_dir='data/imageVal',
            seg_dir='data/labelVal', 
            is_train=False            # Validation uses simpler transforms
        )

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1)

    Args:
        image_dir (str): Directory containing the 3D image volumes
        seg_dir (str): Directory containing the segmentation masks
        transform (Transform, optional): Transform to be applied to the images
        seg_transform (Transform, optional): Transform to be applied to the masks
    """
    def __init__(self,
                 image_dir, 
                 seg_dir, 
                 transform=None, 
                 seg_transform=None):
        super().__init__(image_files=glob(os.path.join(image_dir,"**")),
                         transform = transform,
                         seg_files=glob(os.path.join(seg_dir,"**")),
                         seg_transform=seg_transform)
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.seg_transform = seg_transform

    def __getitem__(self,index):
        return super().__getitem__(index)
    
    @classmethod
    def from_default(cls,image_dir,seg_dir,is_train=True,chunck_size=(64,64,64)):  
        if is_train:
            seg3d_train_imtrans = Compose([LoadImage(image_only=True),
            EnsureChannelFirst(),
            RandSpatialCrop(chunck_size,random_size=False),
            ScaleIntensity(),
            RandRotate90(prob=0.5,spatial_axes=(0,1,2))])    

            seg3d_train_segtrans = Compose([LoadImage(image_only=True),
            EnsureChannelFirst(),
            RandSpatialCrop(chunck_size,random_size=False),
            RandRotate90(prob=0.5,spatial_axes=(0,1,2))])
            return cls(image_dir,seg_dir,transform=seg3d_train_imtrans,seg_transform=seg3d_train_segtrans)
        else:
            seg3d_val_imtrans = Compose([LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity()])

            seg3d_val_segtrans = Compose([LoadImage(image_only=True),
            EnsureChannelFirst()])
            return cls(image_dir,seg_dir,transform=seg3d_val_imtrans,seg_transform=seg3d_val_segtrans)
