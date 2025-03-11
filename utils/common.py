"""
Common utility functions used across the project.
"""

import os
import random
import numpy as np
import torch
import logging
import time
import json
from pathlib import Path

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(output_dir, name='training'):
    """
    Set up logging configuration.
    
    Args:
        output_dir (str): Directory to save log files
        name (str): Name of the log file
        
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}.log")
    
    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def time_function(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func (callable): Function to time
        
    Returns:
        callable: Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def save_json(data, file_path):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        file_path (str): Path to save the JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def ensure_dir(directory):
    """
    Make sure the directory exists.
    
    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
