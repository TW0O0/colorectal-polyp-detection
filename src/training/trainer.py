import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import ColorectalPolypsDataset, get_transforms
from src.models.unet import UNet
from src.training.metrics import dice_coefficient, iou_score
from src.utils.visualization import save_predictions


def train_model(config):
    """
    Train the segmentation model.
    
    Args:
        config (dict): Configuration parameters
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for checkpoints and logs
    os.makedirs(config['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['logs_dir'], exist_ok=True)
    os.makedirs(config['predictions_dir'], exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(config['logs_dir'])
    
    # Create datasets
    train_dataset = ColorectalPolypsDataset(
        images_dir=os.path.join(config['data_dir'], 'train/positive/images'),
        masks_dir=os.path.join(config['data_dir'], 'train/positive/masks'),
        transform=get_transforms('train')
    )
    
    val_dataset = ColorectalPolypsDataset(
        images_dir=os.path.join(config['data_dir'], 'val/positive/images'),
        masks_dir=os.path.join(config['data_dir'], 'val/positive/masks'),
        transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = UNet(
        n_channels=3,
        n_classes=1,
        bilinear=config['bilinear'],
        features=config['features']
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5, 
        verbose=True
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Apply sigmoid to get probabilities
            pred_masks = torch.sigmoid(outputs) > 0.5
            batch_dice = dice_coefficient(pred_masks.float(), masks)
            batch_iou = iou_score(pred_masks.float(), masks)
            
            train_dice += batch_dice.item()
            train_iou += batch_iou.item()
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Update metrics
                val_loss += loss.item()
                
                # Apply sigmoid to get probabilities
                pred_masks = torch.sigmoid(outputs) > 0.5
                batch_dice = dice_coefficient(pred_masks.float(), masks)
                batch_iou = iou_score(pred_masks.float(), masks)
                
                val_dice += batch_dice.item()
                val_iou += batch_iou.item()
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save sample predictions
        if epoch % config['save_pred_interval'] == 0:
            save_predictions(
                model, 
                val_loader, 
                device, 
                config['predictions_dir'], 
                epoch,
                max_samples=8
            )
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, os.path.join(config['checkpoints_dir'], 'best_model.pt'))
            print("Saved best model checkpoint!")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # Save regular checkpoint
        if (epoch + 1) % config['save_checkpoint_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, os.path.join(config['checkpoints_dir'], f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train polyp segmentation model')
    parser.add_argument('--config', type=str, default='configs/unet_config.json',
                        help='Path to configuration JSON file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set default configuration values if not provided
    default_config = {
        'data_dir': 'data/colorectal_data',
        'checkpoints_dir': 'models/checkpoints',
        'logs_dir': 'logs',
        'predictions_dir': 'predictions',
        'batch_size': 16,
        'num_workers': 4,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'features': [64, 128, 256, 512],
        'bilinear': True,
        'save_checkpoint_interval': 10,
        'save_pred_interval': 5,
        'early_stopping_patience': 15
    }
    
    # Update with user-provided configuration
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # Print configuration
    print("Training with configuration:")
    print(json.dumps(config, indent=4))
    
    # Train model
    train_model(config)
