"""
Data loading and preprocessing utilities for TADF dataset
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add path for tadf_dataset import
DATA_DIR = "../../../data_processing"
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR))
if module_path not in sys.path:
    sys.path.append(module_path)

from tadf_dataset import TadfPL


class TadfDataManager:
    """Manages TADF dataset loading and preprocessing."""
    
    def __init__(self, data_dir="../../data_processing"):
        self.data_dir = data_dir
        self.pl_mean = None
        self.pl_std = None
    
    def load_tadf_data(self, force_reload=True, use_H=False):
        """
        Load and split TADF dataset into train/test/val sets.
        
        Returns:
            tuple: (tadf_train, tadf_test, tadf_val)
        """
        tadf_dataset = TadfPL(
            root=os.path.join(self.data_dir, "tadf_pl"), 
            use_H=use_H, 
            force_reload=force_reload
        )

        # Load split files
        splits = {}
        for split in ['test', 'train', 'val']:
            with open(os.path.join(self.data_dir, f"tadf_pl/raw/{split}.csv"), "r") as f:
                splits[split] = [
                    "tadf_" + line.split("|")[0] 
                    for line in f.read().split("\n")[1:-1]
                ]

        # Split dataset
        tadf_train, tadf_test, tadf_val = [], [], []
        for data in tadf_dataset:
            if data.name in splits['test']:
                tadf_test.append(data)
            elif data.name in splits['train']:
                tadf_train.append(data)
            elif data.name in splits['val']:
                tadf_val.append(data)
            else:
                raise ValueError(f"molecule name {data.name} not recognized")

        return tadf_train, tadf_test, tadf_val
    
    def normalize_data_target(self, data_list, pl_mean=None, pl_std=None):
        """
        Normalize target values in the dataset.
        
        Args:
            data_list: List of data objects
            pl_mean: Mean for normalization (calculated if None)
            pl_std: Std for normalization (calculated if None)
            
        Returns:
            tuple: (normalized_data_list, pl_mean, pl_std)
        """
        if pl_mean is None:
            pl_mean = np.mean([data.pl for data in data_list])
        if pl_std is None:
            pl_std = np.std([data.pl for data in data_list])
        
        normalized_data = []
        for data in data_list:
            normalized_data_point = data.clone()
            normalized_data_point.pl = (data.pl - pl_mean) / pl_std
            normalized_data.append(normalized_data_point)
        
        self.pl_mean = pl_mean
        self.pl_std = pl_std
        
        return normalized_data, pl_mean, pl_std
    
    def denormalize(self, pl_value):
        """Denormalize prediction values."""
        if self.pl_mean is None or self.pl_std is None:
            raise ValueError("Normalization parameters not set. Call normalize_data_target first.")
        return pl_value * self.pl_std + self.pl_mean
    
    def get_train_dataloader(self, train_data, batch_size=128, shuffle=True, 
                           augmentation=None, **kwargs):
        """
        Create training dataloader with optional augmentation.
        
        Args:
            train_data: Training dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augmentation: Type of augmentation ('pl' or None)
            
        Returns:
            tuple: (train_loader, pl_mean, pl_std)
        """
        # Apply augmentation if specified
        collate_fn = None
        if augmentation == "pl":
            try:
                from .transforms import augment_pl_with_target_noise
                # augment_pl_with_target_noise expects the data as the first argument,
                # so wrap it into a collate function that DataLoader will call with a batch.
                collate_fn = lambda data_batch: augment_pl_with_target_noise(
                    data_batch, noise_std=kwargs.get("pl_noise", 0.1)
                )
            except ImportError:
                print("Warning: Could not import augmentation function, skipping augmentation")
                collate_fn = None
        
        # Normalize data
        normalized_data, pl_mean, pl_std = self.normalize_data_target(train_data)
        
        train_loader = DataLoader(
            normalized_data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        
        return train_loader, pl_mean, pl_std
    
    def get_eval_dataloader(self, eval_data, pl_mean, pl_std, 
                           batch_size=128, shuffle=False):
        """
        Create evaluation dataloader.
        
        Args:
            eval_data: Evaluation dataset
            pl_mean: Mean for normalization
            pl_std: Std for normalization
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for evaluation
        """
        normalized_data, _, _ = self.normalize_data_target(eval_data, pl_mean, pl_std)
        
        return DataLoader(
            normalized_data,
            batch_size=batch_size,
            shuffle=shuffle,
        )