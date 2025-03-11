"""
Utility functions for model handling, loading, and saving.
"""

import os
import torch
import torch.nn as nn

def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        epoch (int): Current epoch
        metrics (dict): Dictionary of metrics
        path (str): Path to save the checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): The model to load weights into
        optimizer (torch.optim.Optimizer): The optimizer to load state into
        path (str): Path to the checkpoint
        
    Returns:
        tuple: (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    print(f"Checkpoint loaded from {path}")
    return model, optimizer, epoch, metrics

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): The model to count parameters for
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(model):
    """
    Initialize model weights.
    
    Args:
        model (nn.Module): The model to initialize weights for
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def get_lr(optimizer):
    """
    Get the current learning rate from the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer
        
    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
