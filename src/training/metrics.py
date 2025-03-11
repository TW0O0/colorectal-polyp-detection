import torch
import numpy as np


def dice_coefficient(pred, target, smooth=1.0):
    """
    Calculate Dice coefficient
    
    Args:
        pred (torch.Tensor): Predicted binary mask
        target (torch.Tensor): Ground truth binary mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: Dice coefficient
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    """
    Calculate IoU (Intersection over Union) score
    
    Args:
        pred (torch.Tensor): Predicted binary mask
        target (torch.Tensor): Ground truth binary mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: IoU score
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred (torch.Tensor): Predicted binary mask
        target (torch.Tensor): Ground truth binary mask
        
    Returns:
        torch.Tensor: Accuracy
    """
    pred_flat = (pred > 0.5).view(-1)
    target_flat = (target > 0.5).view(-1)
    
    correct = (pred_flat == target_flat).sum().float()
    total = target_flat.numel()
    
    return correct / total


def precision(pred, target, smooth=1.0):
    """
    Calculate precision
    
    Args:
        pred (torch.Tensor): Predicted binary mask
        target (torch.Tensor): Ground truth binary mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: Precision
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    true_positives = (pred_flat * target_flat).sum()
    predicted_positives = pred_flat.sum()
    
    return (true_positives + smooth) / (predicted_positives + smooth)


def recall(pred, target, smooth=1.0):
    """
    Calculate recall
    
    Args:
        pred (torch.Tensor): Predicted binary mask
        target (torch.Tensor): Ground truth binary mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: Recall
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    true_positives = (pred_flat * target_flat).sum()
    actual_positives = target_flat.sum()
    
    return (true_positives + smooth) / (actual_positives + smooth)


def calculate_metrics(pred, target):
    """
    Calculate multiple metrics at once
    
    Args:
        pred (torch.Tensor): Predicted binary mask
        target (torch.Tensor): Ground truth binary mask
        
    Returns:
        dict: Dictionary with metrics
    """
    metrics = {
        'dice': dice_coefficient(pred, target).item(),
        'iou': iou_score(pred, target).item(),
        'accuracy': accuracy(pred, target).item(),
        'precision': precision(pred, target).item(),
        'recall': recall(pred, target).item()
    }
    
    return metrics
