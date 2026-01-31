"""
Simplified K-Fold Cross-Validation Training
Uses environment variables and sequential fold training
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import json
import subprocess
import logging
from collections import defaultdict

sys.path.append(os.getcwd())

from ocr.datasets.kfold_split import KFoldSplitter
from ocr.datasets import get_datasets_by_cfg
import hydra
from omegaconf import OmegaConf

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'
logger = logging.getLogger(__name__)


def setup_kfold_splits(config_path='configs/preset/example.yaml', n_splits=5):
    """Setup K-Fold splits"""
    from ocr.datasets import get_datasets_by_cfg
    
    # Load config
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    
    cfg_dir = str(Path(config_path).parent.resolve())
    
    # Get datasets
    from ocr.datasets.base import OCRDataset
    
    train_image_path = Path('data/datasets/images/train')
    train_ann_path = Path('data/datasets/jsons/train.json')
    
    if not train_image_path.exists() or not train_ann_path.exists():
        print(f"Error: Dataset not found at {train_image_path} or {train_ann_path}")
        return None
    
    train_dataset = OCRDataset(str(train_image_path), str(train_ann_path), transform=None)
    
    # Create K-Fold splitter
    kfold_splitter = KFoldSplitter(n_splits=n_splits, random_state=42)
    
    # Save fold mapping
    kfold_dir = Path('kfold_results')
    kfold_dir.mkdir(exist_ok=True)
    fold_mapping = kfold_splitter.save_fold_mapping(train_dataset.anns, kfold_dir / 'fold_mapping.json')
    
    print(f"\n{'='*60}")
    print(f"K-Fold Split Setup Complete: {n_splits} folds")
    print(f"{'='*60}")
    for fold_name, fold_data in fold_mapping.items():
        print(f"{fold_name}: train={fold_data['train_count']}, val={fold_data['val_count']}")
    
    return fold_mapping


def train_single_fold(fold_idx, n_splits=5):
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}/{n_splits}")
    print(f"{'='*60}")
    
    # Set environment variable for fold
    os.environ['KFOLD_IDX'] = str(fold_idx)
    os.environ['KFOLD_SPLITS'] = str(n_splits)
    
    # Run standard training
    cmd = f"cd {os.getcwd()} && python runners/train.py preset=example wandb=True"
    
    result = os.system(cmd)
    
    if result != 0:
        print(f"Error training fold {fold_idx}")
        return False
    
    return True


def run_kfold_training(n_splits=5):
    """Run complete K-Fold training"""
    print(f"\n{'='*70}")
    print(f"Starting K-Fold Cross-Validation: {n_splits} Folds")
    print(f"Scheduler: CosineAnnealingLR (T_max=20)")
    print(f"Max Epochs: 20")
    print(f"Early Stopping: patience=5")
    print(f"{'='*70}")
    
    # Setup splits
    fold_mapping = setup_kfold_splits(n_splits=n_splits)
    if fold_mapping is None:
        return
    
    results_dir = Path('kfold_results')
    results_dir.mkdir(exist_ok=True)
    
    fold_results = {}
    
    # Train each fold
    for fold_idx in range(n_splits):
        success = train_single_fold(fold_idx, n_splits)
        fold_results[f'fold_{fold_idx}'] = {'success': success, 'timestamp': datetime.now().isoformat()}
    
    # Save results
    results_file = results_dir / 'kfold_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"K-Fold Training Complete!")
    print(f"Results: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Run K-Fold training
    run_kfold_training(n_splits=5)
