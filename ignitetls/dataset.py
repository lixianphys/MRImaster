from monai.data import ImageDataset, CacheDataset
import os
from glob import glob
from monai.transforms import (
    Transform,
    LoadImaged, 
    EnsureChannelFirstd,
    EnsureChannelFirst, 
    ScaleIntensityd, 
    ScaleIntensity,
    Compose, 
    RandRotate90d,
    RandRotate90,
    CenterSpatialCropd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    EnsureTyped,
    Resize,

    )
from torchvision.datasets.folder import make_dataset


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
        img_extensions (str): Extensions of the image files.
    """
    def __init__(self,
                 root:str,
                 transform:Transform|None=None,
                 img_extensions:str=".jpg"):
        list_image_to_label = make_dataset(root,class_to_idx= {x.split("/")[-1]:i for i, x in enumerate(glob(root+'/*'))},extensions=img_extensions)
        image_files = [img_lbl[0] for img_lbl in list_image_to_label]
        labels = [img_lbl[1] for img_lbl in list_image_to_label]
        super().__init__(image_files=image_files,
                         labels = labels,
                         transform=transform)
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self,index):
        return super().__getitem__(index)
    
    @classmethod
    def from_default(cls, root: str, is_train: bool = True, size: tuple[int, int] = (256, 256)):
        if is_train:
            clf2d_train_transform = Compose([
                EnsureChannelFirst(channel_dim=None),
                ScaleIntensity(), 
                Resize(spatial_size=size),
                RandRotate90(prob=0.5)])
            return cls(root,transform=clf2d_train_transform)
        else:  
            clf2d_val_transform = Compose([
                EnsureChannelFirst(channel_dim=None),
                ScaleIntensity(),
                Resize(spatial_size=size)])
            return cls(root,transform=clf2d_val_transform)


class segDataset(CacheDataset):
    """
    Dataset for 3D segmentation tasks.
    This class provides a dataset for 3D medical image segmentation tasks.
    It inherits from MONAI's CacheDataset and provides convenient functionality
    for loading and transforming 3D medical images.

    Example usage:
        # Create training dataset with default transforms
        train_ds = segDataset.from_default(
            image_dir='data/imageTr',  # Directory containing 3D image volumes (e.g. NIfTI files)
            seg_dir='data/labelTr',    # Directory containing segmentation masks
            is_train=True,
            chunk_size=(64, 64, 64)   # Size to resize images to
        )

        # Create validation dataset
        val_ds = segDataset.from_default(
            image_dir='data/imageVal',
            seg_dir='data/labelVal', 
            is_train=False,            # Validation uses simpler transforms
            chunk_size=(64, 64, 64)
        )

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1)

    Args:
        image_dir (str): Directory containing the 3D image volumes
        seg_dir (str): Directory containing the segmentation masks
        transform (Transform, optional): Transform to be applied to both images and masks
        seg_transform (Transform, optional): Additional transform to be applied to masks only
    """
    def __init__(self,
                 image_dir:str, 
                 seg_dir:str, 
                 transform:Transform|None=None, 
                 seg_transform:Transform|None=None):
        super().__init__(data=[{"image":img,"label":seg} for img,seg in zip(sorted(glob(os.path.join(image_dir,"**"))),sorted(glob(os.path.join(seg_dir,"**"))))],transform = transform)
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.seg_transform = seg_transform

    def __getitem__(self,index):
        return super().__getitem__(index)
    
    @classmethod
    def from_default(cls,image_dir:str,seg_dir:str,is_train:bool=True,chunck_size:tuple[int,int,int]=(64,64,64),channel_dim:int=-1):  
        if is_train:
            train_transform = Compose([LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image"],channel_dim=channel_dim),
            EnsureChannelFirstd(keys=["label"],channel_dim=None),  # Need to ensure label has channel first before cropping
            RandSpatialCropd(keys=["image","label"],roi_size=chunck_size,random_size=False),
            ScaleIntensityd(keys="image"),
            RandRotate90d(keys=["image","label"],prob=0.5,spatial_axes=[0,1]),
            EnsureTyped(keys=["image", "label"])])    
            return cls(image_dir,seg_dir,transform=train_transform)
        else:
            val_transform = Compose([LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image"],channel_dim=channel_dim),
            EnsureChannelFirstd(keys=["label"],channel_dim=None), 
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=["image", "label"])])
            return cls(image_dir,seg_dir,transform=val_transform)
