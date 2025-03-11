import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ColorectalPolypsDataset(Dataset):
    """
    Dataset class for colorectal polyp segmentation dataset.
    """
    def __init__(self, 
                 images_dir, 
                 masks_dir=None, 
                 transform=None, 
                 phase='train',
                 is_test=False):
        """
        Args:
            images_dir (str): Path to directory containing images
            masks_dir (str, optional): Path to directory containing masks
            transform (albumentations.Compose, optional): Transformation to apply on images and masks
            phase (str): 'train', 'val', or 'test'
            is_test (bool): Whether this is test data with no masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.phase = phase
        self.is_test = is_test
        
        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.is_test:
            mask_path = os.path.join(self.masks_dir, self.image_files[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Ensure mask is binary (0 or 1)
            mask = mask / 255.0
            mask = (mask > 0.5).astype(np.float32)
        else:
            # For test samples without masks, create a dummy mask
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default transform if none specified
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
            augmented = transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # Add channel dimension
        
        # Get filename without extension for reference
        filename = os.path.splitext(self.image_files[idx])[0]
        
        return {
            'image': image,
            'mask': mask,
            'filename': filename
        }


def get_transforms(phase):
    """
    Return transforms for a specific phase.
    
    Args:
        phase (str): 'train', 'val', or 'test'
        
    Returns:
        albumentations.Compose: Transformation pipeline
    """
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:  # val or test
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
