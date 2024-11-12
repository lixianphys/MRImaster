import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


brat_metadata = {
    "name": "BRATS", 
    "description": "Gliomas segmentation tumour and oedema in on brain images",
    "reference": "https://www.med.upenn.edu/sbia/brats2017.html",
    "licence":"CC-BY-SA 4.0",
    "release":"2.0 04/05/2018",
    "tensorImageSize": "4D",
    "modality": { 
        "0": "FLAIR", 
        "1": "T1w", 
        "2": "t1gd",
        "3": "T2w"
    },  
    "labels": { 
        "0": "background", 
        "1": "edema",
        "2": "non-enhancing tumor",
        "3": "enhancing tumour"
    }
}


class LazyLoadingNiftiDataset(Dataset):
    def __init__(self, image_paths, label_paths, cache_dir, target_shape=(128, 128, 128), transforms=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.cache_dir = cache_dir
        self.target_shape = target_shape
        self.transforms = transforms

        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_paths)

    def load_and_cache_image(self, idx):
        """Load and cache a 4D image and its 3D label if not cached, or load from cache."""
        
        # Define cache paths for the image and label
        img_cache_path = os.path.join(self.cache_dir, f"image_{idx}.npy")
        lbl_cache_path = os.path.join(self.cache_dir, f"label_{idx}.npy")
        
        # Check if the files are already cached
        if os.path.exists(img_cache_path) and os.path.exists(lbl_cache_path):
            # print("Cached images and labels found. Loading from cache")
            # Load from cache
            img_4d = np.load(img_cache_path)
            label_3d = np.load(lbl_cache_path)
        else:
            print("Cached images and labels not found. Loading from nifti files directly and cache to disk")

            # Load the NIfTI files and preprocess
            img_4d = nib.load(self.image_paths[idx]).get_fdata()
            label_3d = nib.load(self.label_paths[idx]).get_fdata()

            # Intensity normalization for the image
            img_4d = (img_4d - np.mean(img_4d)) / np.std(img_4d)
            img_4d = np.clip(img_4d, 0, 1)

            # Ensure label values are within expected range
            label_3d = np.clip(label_3d, 0, 3)

            # Save the preprocessed image and label to cache
            np.save(img_cache_path, img_4d)
            np.save(lbl_cache_path, label_3d)
            print(f"Cached image and label {idx} to disk.")

        return img_4d, label_3d

    def __getitem__(self, idx):
        # Load data from cache or process and cache if needed
        img_4d, label_3d = self.load_and_cache_image(idx)

        # Optionally crop to target shape
        if self.target_shape:
            img_4d = self.center_crop(img_4d, self.target_shape)
            label_3d = self.center_crop(label_3d, self.target_shape)

        # Apply any transformations, if provided
        if self.transforms:
            img_4d, label_3d = self.transforms(img_4d, label_3d)

        # Convert to PyTorch tensors
        img_tensor = torch.tensor(img_4d, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, H, W, D)
        label_tensor = torch.tensor(label_3d, dtype=torch.long)  # (H, W, D)

        return img_tensor, label_tensor

    def center_crop(self, img, target_shape):
        """Crop the center of the image to the target shape."""
        crop_slices = tuple(
            slice((dim - target) // 2, (dim - target) // 2 + target)
            for dim, target in zip(img.shape, target_shape)
        )
        return img[crop_slices]

if __name__ == "__main__":

    # Paths to NIfTI images and labels
    image_paths = ['../data/BRATS_484_img.nii','../data/BRATS_483_img.nii']
    label_paths = ['../data/BRATS_484_lbl.nii','../data/BRATS_483_lbl.nii']
    cache_dir = "../data/cache_dir"
    # Initialize the dataset with caching
    dataset = LazyLoadingNiftiDataset(image_paths=image_paths, label_paths=label_paths, cache_dir=cache_dir)

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Iterate through the DataLoader
    for img_batch, label_batch in dataloader:
        print("Image batch shape:", img_batch.shape)  # Expected shape: (batch_size, 4, H, W, D)
        print("Label batch shape:", label_batch.shape)  # Expected shape: (batch_size, H, W, D)
