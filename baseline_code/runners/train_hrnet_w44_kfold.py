#!/usr/bin/env python3
"""
HRNet-W44 5-Fold Ensemble Training
===================================
Architecture: HRNet-W44 (57M parameters)
Resolution: Progressive (640→960)
Best Single: 96.44% LB (Fold 0, Epoch 10)

Optimal Parameters (Validated):
- lr: 0.00045
- weight_decay: 0.000082
- T_max: 20
- eta_min: 0.000008
- early_stopping_patience: 5

Expected Performance: 96.5-96.7% LB (5-fold ensemble)

Usage:
    python runners/train_hrnet_w44_kfold.py

Timeline (approx 7.5 hours total):
    - Fold 0: ~90 min (already trained, 96.44%)
    - Fold 1: ~90 min
    - Fold 2: ~90 min
    - Fold 3: ~90 min
    - Fold 4: ~90 min
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import hydra
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.datasets.kfold_split import load_fold_split

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train_kfold(config):
    """
    Train HRNet-W44 5-fold ensemble with optimal parameters.
    """
    
    fold_results = {}
    fold_mapping = load_fold_split()
    
    for fold_idx in range(5):
        print(f"\n{'='*80}")
        print(f"Training Fold {fold_idx}/4")
        print(f"{'='*80}\n")
        
        # Set seed for reproducibility
        pl.seed_everything(config.get("seed", 42) + fold_idx, workers=True)
        
        # Get model and data modules
        model_module, data_module = get_pl_modules_by_cfg(config)
        
        # Setup data for this fold
        train_indices = fold_mapping[f"fold_{fold_idx}"]["train"]
        val_indices = fold_mapping[f"fold_{fold_idx}"]["val"]
        
        data_module.setup_fold(train_indices, val_indices)
        
        # Logger setup
        exp_name = f"hrnet_w44_fold{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        if config.get("wandb"):
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(
                project=config.get('wandb_project', 'ocr-hrnet-w44-kfold'),
                name=f"HRNet-W44 Fold {fold_idx}",
                config=dict(config),
                log_model=True,
                tags=["hrnet-w44", "kfold", f"fold-{fold_idx}", "ensemble"],
            )
        else:
            logger = True
        
        # Callbacks
        checkpoint_dir = f"./checkpoints/hrnet_w44_fold{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best_model",
                monitor="val_metric",
                mode="max",
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_metric",
                mode="max",
                patience=5,  # Optimal value from experiments
                verbose=True,
            ),
        ]
        
        # Trainer configuration
        trainer = pl.Trainer(
            max_epochs=20,  # Optimal from experiments
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            precision="32-bit",
            enable_progress_bar=True,
        )
        
        # Train
        trainer.fit(model_module, data_module)
        
        # Test (validation set as test)
        test_results = trainer.test(model_module, data_module)
        
        # Record fold results
        if test_results:
            fold_results[f"fold_{fold_idx}"] = test_results[0]
        
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx} Complete")
        print(f"{'='*80}\n")
    
    # Summary
    print("\n" + "="*80)
    print("HRNet-W44 5-Fold Ensemble Training Complete!")
    print("="*80)
    print(f"\nFold Results:")
    for fold, results in fold_results.items():
        print(f"  {fold}: {results}")
    
    # Calculate average
    if fold_results:
        avg_metric = sum(r.get('val_metric', 0) for r in fold_results.values()) / len(fold_results)
        print(f"\nAverage Metric: {avg_metric:.4f}")
        print(f"Expected LB: 96.5-96.7%")
    
    print("="*80 + "\n")
    
    # Save results
    results_file = f"hrnet_w44_kfold_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=2)
    print(f"Results saved to: {results_file}\n")


if __name__ == "__main__":
    train_kfold()
