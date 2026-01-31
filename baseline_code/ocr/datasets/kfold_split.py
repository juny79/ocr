"""
K-Fold Cross-Validation Split Module

This module provides utilities for splitting datasets into K-Fold splits
for cross-validation training.
"""

import numpy as np
import json
from pathlib import Path
from collections import OrderedDict
import logging
from typing import Dict, Tuple, Iterator

logger = logging.getLogger(__name__)


def get_fold_indices(n_samples: int, fold_idx: int, n_splits: int = 5, 
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get train/val indices for a specific fold
    
    Args:
        n_samples: Total number of samples
        fold_idx: Current fold index (0 to n_splits-1)
        n_splits: Total number of folds
        random_state: Random seed
    
    Returns:
        (train_indices, val_indices)
    """
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    fold_size = n_samples // n_splits
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < n_splits - 1 else n_samples
    
    val_indices = indices[val_start:val_end]
    train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
    
    return train_indices, val_indices


class KFoldSplitter:
    """K-Fold Dataset Splitter for Cross-Validation"""

    def __init__(self, n_splits=5, random_state=42):
        """
        Initialize K-Fold Splitter
        
        Args:
            n_splits (int): Number of folds. Default: 5
            random_state (int): Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def split(self, dataset_annotations):
        """
        Split dataset into K-Folds
        
        Args:
            dataset_annotations (dict or OrderedDict): Annotations dictionary 
                                                       with filename as key
        
        Yields:
            tuple: (train_indices, val_indices) for each fold
        """
        # Get all keys (filenames)
        all_keys = list(dataset_annotations.keys())
        n_samples = len(all_keys)
        
        # Shuffle indices
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        # Split into K folds
        fold_size = n_samples // self.n_splits
        
        for fold_idx in range(self.n_splits):
            # Calculate validation indices for this fold
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < self.n_splits - 1 else n_samples
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            yield fold_idx, train_indices, val_indices

    def get_fold_annotations(self, dataset_annotations, fold_idx, split='train'):
        """
        Get annotations for a specific fold and split
        
        Args:
            dataset_annotations (dict): Annotations dictionary
            fold_idx (int): Fold index
            split (str): 'train' or 'val'
        
        Returns:
            OrderedDict: Filtered annotations for the fold
        """
        all_keys = list(dataset_annotations.keys())
        indices = np.arange(len(all_keys))
        self.rng.seed(self.random_state)
        self.rng.shuffle(indices)
        
        fold_size = len(all_keys) // self.n_splits
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < self.n_splits - 1 else len(all_keys)
        
        if split == 'val':
            selected_indices = indices[val_start:val_end]
        else:  # train
            selected_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        selected_keys = [all_keys[i] for i in selected_indices]
        fold_annotations = OrderedDict({k: dataset_annotations[k] for k in selected_keys})
        
        return fold_annotations

    def save_fold_mapping(self, dataset_annotations, output_path):
        """
        Save K-Fold split mapping to JSON file
        
        Args:
            dataset_annotations (dict): Original annotations
            output_path (str or Path): Output file path
        """
        all_keys = list(dataset_annotations.keys())
        indices = np.arange(len(all_keys))
        self.rng.seed(self.random_state)
        self.rng.shuffle(indices)
        
        fold_mapping = {}
        fold_size = len(all_keys) // self.n_splits
        
        for fold_idx in range(self.n_splits):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < self.n_splits - 1 else len(all_keys)
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            val_keys = [all_keys[i] for i in val_indices]
            train_keys = [all_keys[i] for i in train_indices]
            
            fold_mapping[f'fold_{fold_idx}'] = {
                'train': train_keys,
                'val': val_keys,
                'train_count': len(train_keys),
                'val_count': len(val_keys)
            }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(fold_mapping, f, indent=2)
        
        logger.info(f"K-Fold mapping saved to {output_path}")
        return fold_mapping


class KFoldDataset:
    """K-Fold aware Dataset wrapper"""
    
    def __init__(self, base_dataset, fold_idx, split='train', kfold_splitter=None):
        """
        Initialize K-Fold Dataset
        
        Args:
            base_dataset: Original dataset with annotations
            fold_idx (int): Current fold index
            split (str): 'train' or 'val'
            kfold_splitter (KFoldSplitter): KFoldSplitter instance
        """
        if kfold_splitter is None:
            kfold_splitter = KFoldSplitter(n_splits=5)
        
        self.base_dataset = base_dataset
        self.fold_idx = fold_idx
        self.split = split
        self.kfold_splitter = kfold_splitter
        
        # Get filtered annotations for this fold
        self.anns = self.kfold_splitter.get_fold_annotations(
            base_dataset.anns, 
            fold_idx, 
            split
        )
        
        # Copy other attributes
        self.image_path = base_dataset.image_path
        self.transform = base_dataset.transform
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        # Get image filename from filtered annotations
        image_filename = list(self.anns.keys())[idx]
        
        # Reconstruct item similar to original dataset
        from PIL import Image
        import numpy as np
        image = Image.open(self.image_path / image_filename).convert('RGB')
        
        # Handle EXIF rotation (if needed)
        exif = image.getexif()
        if exif:
            from ocr.datasets.base import OCRDataset
            orientation_key = 274
            if orientation_key in exif:
                image = OCRDataset.rotate_image(image, exif[orientation_key])
        
        org_shape = image.size
        
        from collections import OrderedDict
        item = OrderedDict(image=image, image_filename=image_filename, shape=org_shape)
        
        # Get annotations (polygons)
        polygons = self.anns[image_filename] or None
        
        # Apply transform (similar to base.py)
        if self.transform is None:
            raise ValueError("Transform function is a required value.")
        
        # Image transform with polygons
        transformed = self.transform(image=np.array(image), polygons=polygons)
        item.update(image=transformed['image'],
                    polygons=transformed['polygons'],
                    inverse_matrix=transformed['inverse_matrix'])
        
        return item
