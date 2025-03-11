import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def save_predictions(model, data_loader, device, save_dir, epoch, max_samples=8):
    """
    Generate and save predictions from the model.
    
    Args:
        model (nn.Module): Trained model
        data_loader (DataLoader): DataLoader with validation/test data
        device (torch.device): Device to run inference on
        save_dir (str): Directory to save images
        epoch (int): Current epoch number
        max_samples (int): Maximum number of samples to visualize
    """
    model.eval()
    
    # Create directory for this epoch
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Get a batch of data
    batch = next(iter(data_loader))
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    filenames = batch['filename']
    
    # Limit number of samples
    num_samples = min(max_samples, images.size(0))
    images = images[:num_samples]
    masks = masks[:num_samples]
    filenames = filenames[:num_samples]
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
    
    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    images = images.transpose(0, 2, 3, 1)
    images = images * std + mean
    images = np.clip(images, 0, 1)
    
    # Plot and save each sample
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(images[i])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(images[i])
        mask_display = np.squeeze(masks[i])
        axes[1].imshow(mask_display, alpha=0.5, cmap='Reds')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(images[i])
        pred_display = np.squeeze(preds[i])
        axes[2].imshow(pred_display, alpha=0.5, cmap='Blues')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # Probability map
        prob_map = np.squeeze(probs[i])
        axes[3].imshow(prob_map, cmap='viridis')
        axes[3].
