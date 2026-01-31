"""
K-Fold Cross-Validation Training Script

This script performs K-Fold cross-validation training with the OCR model.
- Splits training data into K folds
- Trains model on each fold
- Aggregates metrics across folds
- Saves fold-specific checkpoints
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from collections import defaultdict, OrderedDict
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
from ocr.datasets.kfold_split import KFoldSplitter, KFoldDataset
from ocr.datasets import get_datasets_by_cfg

logger = logging.getLogger(__name__)
CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


class KFoldTrainer:
    """K-Fold Cross-Validation Trainer"""
    
    def __init__(self, config, n_splits=5):
        """
        Initialize K-Fold Trainer
        
        Args:
            config: Hydra configuration
            n_splits (int): Number of folds
        """
        self.config = config
        self.n_splits = n_splits
        self.kfold_splitter = KFoldSplitter(n_splits=n_splits, random_state=42)
        self.fold_results = defaultdict(list)
        self.fold_checkpoints = {}
        
        # Create K-Fold results directory
        self.kfold_results_dir = Path('kfold_results')
        self.kfold_results_dir.mkdir(exist_ok=True)
        
        self.fold_mapping_file = self.kfold_results_dir / 'fold_mapping.json'
        
    def prepare_fold_splits(self):
        """Prepare K-Fold splits from training dataset"""
        # Load original datasets to get annotations
        datasets = get_datasets_by_cfg(self.config.datasets)
        train_dataset = datasets['train']
        
        # Save fold mapping
        self.kfold_splitter.save_fold_mapping(
            train_dataset.anns,
            self.fold_mapping_file
        )
        
        logger.info(f"K-Fold splits created: {self.n_splits} folds")
        logger.info(f"Total training samples: {len(train_dataset)}")
        
    def train_fold(self, fold_idx):
        """
        Train model on a single fold
        
        Args:
            fold_idx (int): Fold index (0 to n_splits-1)
        
        Returns:
            dict: Fold results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold_idx + 1}/{self.n_splits}")
        logger.info(f"{'='*60}")
        
        # Set seed for reproducibility
        pl.seed_everything(self.config.get("seed", 42), workers=True)
        
        # Create model
        model_module, data_module = get_pl_modules_by_cfg(self.config)
        
        # Modify datasets for K-Fold
        datasets = get_datasets_by_cfg(self.config.datasets)
        
        # Create K-Fold aware datasets
        train_dataset = KFoldDataset(
            datasets['train'],
            fold_idx=fold_idx,
            split='train',
            kfold_splitter=self.kfold_splitter
        )
        val_dataset = KFoldDataset(
            datasets['train'],
            fold_idx=fold_idx,
            split='val',
            kfold_splitter=self.kfold_splitter
        )
        
        # Update data module datasets
        data_module.dataset['train'] = train_dataset
        data_module.dataset['val'] = val_dataset
        model_module.dataset['train'] = train_dataset
        model_module.dataset['val'] = val_dataset
        
        logger.info(f"Fold {fold_idx}: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        # Setup logger
        if self.config.get("wandb"):
            from lightning.pytorch.loggers import WandbLogger
            import wandb
            
            exp_name = f"kfold_fold{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            logger_obj = WandbLogger(
                project=self.config.project_name,
                name=exp_name,
                config=dict(self.config),
                log_model=True,
                tags=["kfold", "baseline", "dbnet", f"fold{fold_idx}"],
            )
        else:
            from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
            logger_obj = TensorBoardLogger(
                save_dir=self.config.log_dir,
                name=f"kfold_fold{fold_idx}",
                version=datetime.now().strftime('%Y%m%d_%H%M'),
                default_hp_metric=False,
            )
        
        # Setup checkpoint saving
        fold_checkpoint_dir = self.kfold_results_dir / f'fold_{fold_idx}' / 'checkpoints'
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
        
        # Create trainer for this fold
        trainer = pl.Trainer(
            **self.config.trainer,
            logger=logger_obj,
            callbacks=callbacks,
        )
        
        # Train
        trainer.fit(
            model_module,
            data_module,
            ckpt_path=self.config.get("resume", None),
        )
        
        # Get best checkpoint path
        best_ckpt = callbacks[1].best_model_path if callbacks[1].best_model_path else None
        self.fold_checkpoints[fold_idx] = best_ckpt
        
        logger.info(f"Fold {fold_idx} best checkpoint: {best_ckpt}")
        
        return {
            'fold_idx': fold_idx,
            'best_checkpoint': best_ckpt,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
        }
    
    def run(self):
        """Run complete K-Fold cross-validation"""
        logger.info("Starting K-Fold Cross-Validation")
        logger.info(f"Configuration: {self.n_splits}-Fold CV")
        
        # Prepare fold splits
        self.prepare_fold_splits()
        
        # Train on each fold
        fold_results = []
        for fold_idx in range(self.n_splits):
            try:
                result = self.train_fold(fold_idx)
                fold_results.append(result)
            except Exception as e:
                logger.error(f"Error training fold {fold_idx}: {e}")
                raise
        
        # Save results
        results_file = self.kfold_results_dir / 'kfold_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'n_splits': self.n_splits,
                'fold_results': fold_results,
                'fold_checkpoints': {k: str(v) for k, v in self.fold_checkpoints.items()},
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        logger.info(f"\nK-Fold Cross-Validation Complete!")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Fold checkpoints: {self.fold_checkpoints}")
        
        return fold_results


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train_kfold(config):
    """
    K-Fold Cross-Validation Training
    
    Args:
        config: Hydra configuration
    """
    # Create K-Fold trainer
    kfold_trainer = KFoldTrainer(config, n_splits=5)
    
    # Run K-Fold training
    results = kfold_trainer.run()
    
    logger.info("\n" + "="*60)
    logger.info("K-Fold Training Summary")
    logger.info("="*60)
    for result in results:
        logger.info(f"Fold {result['fold_idx']}: "
                   f"Train={result['train_size']}, "
                   f"Val={result['val_size']}")


if __name__ == "__main__":
    train_kfold()
