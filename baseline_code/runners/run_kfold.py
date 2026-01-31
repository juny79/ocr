#!/usr/bin/env python
"""
K-Fold Cross-Validation Training Manager

Simple manager that:
1. Splits data into K folds using KFoldSplitter
2. Modifies dataset configs dynamically for each fold
3. Runs training pipeline 5 times with different fold configs
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

sys.path.append(os.getcwd())

from ocr.datasets.kfold_split import KFoldSplitter, get_fold_indices
from ocr.datasets.base import OCRDataset


def setup_fold_splits(n_splits=5):
    """Create K-Fold split mapping"""
    print(f"\n{'='*70}")
    print(f"Setting up {n_splits}-Fold Cross-Validation")
    print(f"{'='*70}")
    
    # Load training dataset (absolute paths)
    base_path = Path('/data/ephemeral/home')
    train_image_path = base_path / 'data/datasets/images/train'
    train_ann_path = base_path / 'data/datasets/jsons/train.json'
    
    if not train_image_path.exists() or not train_ann_path.exists():
        raise FileNotFoundError(f"Dataset not found: {train_image_path} or {train_ann_path}")
    
    train_dataset = OCRDataset(str(train_image_path), str(train_ann_path), transform=None)
    n_samples = len(train_dataset)
    all_filenames = list(train_dataset.anns.keys())
    
    print(f"Total training samples: {n_samples}")
    
    # Create splits mapping
    fold_mapping = {}
    for fold_idx in range(n_splits):
        train_indices, val_indices = get_fold_indices(n_samples, fold_idx, n_splits, random_state=42)
        
        train_files = [all_filenames[i] for i in train_indices]
        val_files = [all_filenames[i] for i in val_indices]
        
        fold_mapping[fold_idx] = {
            'train_files': train_files,
            'val_files': val_files,
            'train_count': len(train_files),
            'val_count': len(val_files),
        }
        
        print(f"Fold {fold_idx}: train={len(train_files)}, val={len(val_files)}")
    
    # Save mapping
    kfold_dir = Path('kfold_results')
    kfold_dir.mkdir(exist_ok=True)
    
    mapping_file = kfold_dir / 'fold_mapping.json'
    with open(mapping_file, 'w') as f:
        # Convert to JSON serializable format
        json_mapping = {}
        for fold_idx, data in fold_mapping.items():
            json_mapping[str(fold_idx)] = data
        json.dump(json_mapping, f, indent=2)
    
    print(f"✓ Fold mapping saved to {mapping_file}\n")
    return fold_mapping


def create_fold_datasets(fold_idx, fold_mapping, n_splits=5):
    """
    Create train/val datasets for specific fold by filtering annotations
    """
    fold_data = fold_mapping[fold_idx]
    train_files = set(fold_data['train_files'])
    val_files = set(fold_data['val_files'])
    
    # Load original datasets (absolute paths)
    base_path = Path('/data/ephemeral/home')
    train_image_path = base_path / 'data/datasets/images/train'
    train_ann_path = base_path / 'data/datasets/jsons/train.json'
    val_image_path = base_path / 'data/datasets/images/val'
    val_ann_path = base_path / 'data/datasets/jsons/val.json'
    
    # Create fold-specific annotation files
    kfold_dir = Path('kfold_results') / f'fold_{fold_idx}'
    kfold_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    
    # Process training annotations
    with open(train_ann_path, 'r') as f:
        train_annotations = json.load(f)
    
    filtered_train_anns = {
        'images': {k: v for k, v in train_annotations['images'].items() if k in train_files}
    }
    
    train_fold_ann_path = kfold_dir / 'train.json'
    with open(train_fold_ann_path, 'w') as f:
        json.dump(filtered_train_anns, f)
    
    # Process validation annotations (use as val set for this fold)
    with open(val_ann_path, 'r') as f:
        val_annotations = json.load(f)
    
    # Save as-is (use full val set as validation for fold)
    val_fold_ann_path = kfold_dir / 'val.json'
    with open(val_fold_ann_path, 'w') as f:
        json.dump(val_annotations, f)
    
    return {
        'train_ann': str(train_fold_ann_path),
        'val_ann': str(val_fold_ann_path),
        'train_img': str(train_image_path),
        'val_img': str(val_image_path),
    }


def train_fold(fold_idx, fold_paths, n_splits=5):
    """Train model on a single fold"""
    print(f"\n{'='*70}")
    print(f"Training Fold {fold_idx + 1}/{n_splits}")
    print(f"{'='*70}")
    
    wandb_api_key = os.environ.get('WANDB_API_KEY', '')
    
    # Set environment variables for this fold
    env = os.environ.copy()
    env['KFOLD_IDX'] = str(fold_idx)
    env['KFOLD_N_SPLITS'] = str(n_splits)
    env['WANDB_API_KEY'] = wandb_api_key
    
    # Run training with standard script
    cmd = [
        sys.executable,
        'runners/train.py',
        'preset=example',
        'wandb=True',
    ]
    
    # Additional config overrides for K-Fold dataset paths
    cmd.extend([
        f"datasets.train_dataset.annotation_path={fold_paths['train_ann']}",
        f"datasets.val_dataset.annotation_path={fold_paths['val_ann']}",
    ])
    
    try:
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())
        return result.returncode == 0
    except Exception as e:
        print(f"Error training fold {fold_idx}: {e}")
        return False


def run_kfold_training(n_splits=5):
    """Run complete K-Fold cross-validation"""
    print(f"\n{'#'*70}")
    print(f"# K-FOLD CROSS-VALIDATION TRAINING")
    print(f"# {n_splits} Folds | T_max=20 | Early Stopping (patience=5)")
    print(f"# Scheduler: CosineAnnealingLR | Max Epochs: 20")
    print(f"{'#'*70}")
    
    # Step 1: Setup fold splits
    fold_mapping = setup_fold_splits(n_splits)
    
    # Step 2: Train each fold
    fold_results = {}
    
    for fold_idx in range(n_splits):
        # Create fold-specific datasets
        fold_paths = create_fold_datasets(fold_idx, fold_mapping, n_splits)
        
        # Train
        success = train_fold(fold_idx, fold_paths, n_splits)
        
        fold_results[fold_idx] = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
        }
        
        if not success:
            print(f"Warning: Fold {fold_idx} training may have failed")
    
    # Step 3: Save results summary
    summary = {
        'n_splits': n_splits,
        'total_samples': sum(fold_mapping[0]['train_count'] + fold_mapping[0]['val_count'] 
                           for _ in range(n_splits)),
        'timestamp': datetime.now().isoformat(),
        'fold_results': fold_results,
    }
    
    results_file = Path('kfold_results') / 'kfold_training_summary.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ K-Fold Cross-Validation Training Complete!")
    print(f"  Results: {results_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_kfold_training(n_splits=5)
