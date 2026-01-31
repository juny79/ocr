#!/usr/bin/env python
"""
Simple K-Fold Split Generator
Creates 5 separate train/val annotation files for K-Fold CV
"""

import json
import numpy as np
from pathlib import Path
from collections import OrderedDict


def create_kfold_splits(n_splits=5):
    """Create K-Fold annotation splits"""
    base_path = Path('/data/ephemeral/home')
    train_ann_path = base_path / 'data/datasets/jsons/train.json'
    
    # Load original training annotations
    with open(train_ann_path, 'r') as f:
        train_data = json.load(f)
    
    all_images = list(train_data['images'].keys())
    n_samples = len(all_images)
    
    print(f"Total training samples: {n_samples}")
    print(f"Creating {n_splits} folds...\n")
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Create fold directories and annotation files
    kfold_dir = base_path / 'data' / 'datasets' / 'jsons' / 'kfold'
    kfold_dir.mkdir(parents=True, exist_ok=True)
    
    fold_info = []
    
    for fold_idx in range(n_splits):
        # Calculate validation indices for this fold
        fold_size = n_samples // n_splits
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_splits - 1 else n_samples
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Get filenames
        train_files = [all_images[i] for i in train_indices]
        val_files = [all_images[i] for i in val_indices]
        
        # Create train annotation for this fold
        fold_train_data = {
            'images': OrderedDict({k: train_data['images'][k] for k in train_files})
        }
        
        # Create val annotation for this fold
        fold_val_data = {
            'images': OrderedDict({k: train_data['images'][k] for k in val_files})
        }
        
        # Save files
        fold_train_path = kfold_dir / f'fold{fold_idx}_train.json'
        fold_val_path = kfold_dir / f'fold{fold_idx}_val.json'
        
        with open(fold_train_path, 'w') as f:
            json.dump(fold_train_data, f, indent=2)
        
        with open(fold_val_path, 'w') as f:
            json.dump(fold_val_data, f, indent=2)
        
        fold_info.append({
            'fold': fold_idx,
            'train_count': len(train_files),
            'val_count': len(val_files),
            'train_file': str(fold_train_path),
            'val_file': str(fold_val_path),
        })
        
        print(f"Fold {fold_idx}: train={len(train_files)}, val={len(val_files)}")
        print(f"  Train: {fold_train_path}")
        print(f"  Val:   {fold_val_path}")
    
    # Save fold info
    info_file = kfold_dir / 'fold_info.json'
    with open(info_file, 'w') as f:
        json.dump(fold_info, f, indent=2)
    
    print(f"\n✓ K-Fold splits created successfully!")
    print(f"  Info: {info_file}")
    
    return fold_info


if __name__ == '__main__':
    create_kfold_splits(n_splits=5)
