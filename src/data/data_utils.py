"""
Utility functions for data handling and transformations.
"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def get_transform(config):
    """
    Create transformations based on configuration.
    
    Args:
        config (dict): Configuration dictionary with transformation parameters
        
    Returns:
        callable: Transformation function
    """
    def transform(image, mask):
        # Convert to PIL Images if they are not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.uint8(mask))
        
        # Resize
        if "resize" in config:
            image = image.resize(config["resize"], Image.BILINEAR)
            mask = mask.resize(config["resize"], Image.NEAREST)
        
        # Random horizontal flip
        if "horizontal_flip" in config and random.random() < config["horizontal_flip"]:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if "vertical_flip" in config and random.random() < config["vertical_flip"]:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        if "rotation" in config:
            angle = random.uniform(-config["rotation"], config["rotation"])
            image = TF.rotate(image, angle, TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, TF.InterpolationMode.NEAREST)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        
        # Random brightness and contrast adjustments
        if "brightness" in config:
            brightness_factor = 1.0 + random.uniform(-config["brightness"], config["brightness"])
            image = TF.adjust_brightness(image, brightness_factor)
        
        if "contrast" in config:
            contrast_factor = 1.0 + random.uniform(-config["contrast"], config["contrast"])
            image = TF.adjust_contrast(image, contrast_factor)
        
        # Normalize
        if "normalize" in config:
            image = TF.normalize(
                image, 
                mean=config["normalize"]["mean"], 
                std=config["normalize"]["std"]
            )
        
        return image, mask
    
    return transform

def denormalize(image, mean, std):
    """
    Denormalize an image tensor.
    
    Args:
        image (torch.Tensor): Normalized image tensor
        mean (list): Mean values used for normalization
        std (list): Standard deviation values used for normalization
        
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return image * std + mean

def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(image_path).convert("RGB")

def load_mask(mask_path):
    """
    Load a mask from file.
    
    Args:
        mask_path (str): Path to the mask file
        
    Returns:
        PIL.Image: Loaded mask as grayscale
    """
    return Image.open(mask_path).convert("L")
