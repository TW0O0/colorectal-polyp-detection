#!/usr/bin/env python3
"""
Main script to train and evaluate the colorectal polyp detection model.
"""

import os
import argparse
import torch
import json
from datetime import datetime

from src.data.dataset import PolypDataset
from src.models.unet import UNet
from src.training.trainer import Trainer
from src.utils.visualization import save_predictions

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train UNet for polyp segmentation')
    parser.add_argument('--config', type=str, default='configs/unet_config.json',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing the processed dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = PolypDataset(
        images_dir=os.path.join(args.data_dir, 'train', 'images'),
        masks_dir=os.path.join(args.data_dir, 'train', 'masks'),
        transform=config['train_transform']
    )
    
    val_dataset = PolypDataset(
        images_dir=os.path.join(args.data_dir, 'val', 'images'),
        masks_dir=os.path.join(args.data_dir, 'val', 'masks'),
        transform=config['val_transform']
    )
    
    # Create model
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        init_features=config['model']['init_features']
    ).to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['num_epochs'],
        output_dir=output_dir
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on test set
    test_dataset = PolypDataset(
        images_dir=os.path.join(args.data_dir, 'test', 'images'),
        masks_dir=os.path.join(args.data_dir, 'test', 'masks'),
        transform=config['test_transform']
    )
    
    test_metrics = trainer.evaluate(test_dataset)
    print(f"Test metrics: {test_metrics}")
    
    # Save predictions on test set
    save_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        output_dir=os.path.join(output_dir, 'test_predictions'),
        num_samples=10
    )
    
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
