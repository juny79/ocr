"""
Simplified K-Fold Cross-Validation Training Script

This script performs K-Fold cross-validation by:
1. Running standard training multiple times with different fold splits
2. Using configuration-based fold selection
3. Tracking metrics for each fold
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import logging

import lightning.pytorch as pl
import hydra
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.getcwd())

from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.datasets.kfold_split import KFoldSplitter

logger = logging.getLogger(__name__)
CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


def create_fold_config_override(fold_idx, n_splits=5):
    """Create config override for specific fold"""
    return f"fold_idx={fold_idx},n_splits={n_splits}"


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train(config):
    """
    K-Fold Cross-Validation Training
    
    Args:
        config: Hydra configuration
    """
    # Check if fold_idx is provided in config
    fold_idx = config.get("fold_idx", None)
    n_splits = config.get("n_splits", 5)
    
    if fold_idx is None:
        # Run all folds sequentially
        logger.info("="*60)
        logger.info("Starting K-Fold Cross-Validation Training")
        logger.info(f"Configuration: {n_splits}-Fold CV")
        logger.info("="*60)
        
        fold_results = defaultdict(dict)
        all_checkpoints = {}
        
        # Create K-Fold splitter
        kfold_splitter = KFoldSplitter(n_splits=n_splits, random_state=42)
        
        # Load base datasets
        from ocr.datasets import get_datasets_by_cfg
        datasets = get_datasets_by_cfg(config.datasets)
        train_dataset = datasets['train']
        
        # Save fold mapping
        kfold_dir = Path('kfold_results')
        kfold_dir.mkdir(exist_ok=True)
        kfold_splitter.save_fold_mapping(train_dataset.anns, kfold_dir / 'fold_mapping.json')
        
        # Train each fold
        for fold_num in range(n_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Fold {fold_num + 1}/{n_splits}")
            logger.info(f"{'='*60}")
            
            # Recursively call with fold_idx set
            os.system(f"cd {os.getcwd()} && "
                     f"export WANDB_API_KEY='{os.environ.get('WANDB_API_KEY')}' && "
                     f"python runners/train_kfold_simple.py preset=example wandb=True "
                     f"fold_idx={fold_num} n_splits={n_splits}")
        
        logger.info("\n" + "="*60)
        logger.info("K-Fold Cross-Validation Complete!")
        logger.info("="*60)
        
    else:
        # Train single fold
        logger.info(f"\nTraining Single Fold: {fold_idx}/{n_splits}")
        
        pl.seed_everything(config.get("seed", 42), workers=True)
        
        # Get model and data modules
        model_module, data_module = get_pl_modules_by_cfg(config)
        
        # Modify datasets for K-Fold split
        from ocr.datasets.kfold_split import KFoldDataset
        from ocr.datasets import get_datasets_by_cfg
        
        kfold_splitter = KFoldSplitter(n_splits=n_splits, random_state=42)
        datasets = get_datasets_by_cfg(config.datasets)
        
        # Create fold-specific datasets
        train_dataset_kfold = KFoldDataset(
            datasets['train'],
            fold_idx=fold_idx,
            split='train',
            kfold_splitter=kfold_splitter
        )
        val_dataset_kfold = KFoldDataset(
            datasets['train'],
            fold_idx=fold_idx,
            split='val',
            kfold_splitter=kfold_splitter
        )
        
        # Update data module
        data_module.dataset['train'] = train_dataset_kfold
        data_module.dataset['val'] = val_dataset_kfold
        model_module.dataset['train'] = train_dataset_kfold
        model_module.dataset['val'] = val_dataset_kfold
        
        logger.info(f"Fold {fold_idx}: Train={len(train_dataset_kfold)}, "
                   f"Val={len(val_dataset_kfold)}")
        
        # Setup logger
        if config.get("wandb"):
            from lightning.pytorch.loggers import WandbLogger
            import wandb
            
            exp_name = f"kfold_fold{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            logger_obj = WandbLogger(
                project=config.project_name,
                name=exp_name,
                config=dict(config),
                log_model=True,
                tags=["kfold", "baseline", "dbnet", f"fold{fold_idx}"],
            )
        else:
            from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
            logger_obj = TensorBoardLogger(
                save_dir=config.log_dir,
                name=f"kfold_fold{fold_idx}",
                version=datetime.now().strftime('%Y%m%d_%H%M'),
                default_hp_metric=False,
            )
        
        # Setup checkpoint saving
        fold_checkpoint_dir = Path('kfold_results') / f'fold_{fold_idx}' / 'checkpoints'
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                dirpath=str(fold_checkpoint_dir),
                save_top_k=3,
                monitor='val/loss',
                mode='min',
                filename='fold_{epoch:02d}-{step}',
            ),
            EarlyStopping(
                monitor='val/loss',
                patience=5,
                mode='min',
                verbose=True,
                check_finite=True,
            ),
        ]
        
        # Create trainer
        trainer = pl.Trainer(
            **config.trainer,
            logger=logger_obj,
            callbacks=callbacks,
        )
        
        # Train
        trainer.fit(
            model_module,
            data_module,
            ckpt_path=config.get("resume", None),
        )
        
        # Test on validation set
        trainer.test(model_module, data_module)
        
        logger.info(f"Fold {fold_idx} training complete!")


if __name__ == "__main__":
    train()
