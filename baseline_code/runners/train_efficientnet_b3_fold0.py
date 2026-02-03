#!/usr/bin/env python
"""
EfficientNet-B3 Hybrid Training Script with Progressive Resolution
- Mixed Precision Training (FP16)
- Early Stopping (patience=5, monitor='val/hmean')
- Progressive Resolution: 640px (epoch 0-3) → 960px (epoch 4+)
- Single Fold (Fold 0) for quick validation
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr.lightning_modules import get_pl_modules_by_cfg, OCRPLModule
from ocr.datasets import get_datasets_by_cfg, DBCollateFN
from torch.utils.data import DataLoader
import json


class ProgressiveResolutionCallback(pl.Callback):
    """Progressive resolution switching: 640px → 960px at epoch 4"""
    
    def __init__(self, switch_epoch=4, initial_size=640, target_size=960):
        super().__init__()
        self.switch_epoch = switch_epoch
        self.initial_size = initial_size
        self.target_size = target_size
        self.switched = False
        
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        # Switch to higher resolution at specified epoch
        if current_epoch >= self.switch_epoch and not self.switched:
            print(f"\n{'='*60}")
            print(f"🔄 Epoch {current_epoch}: Switching resolution")
            print(f"   {self.initial_size}px → {self.target_size}px")
            print(f"{'='*60}\n")
            
            # Update dataset transforms
            for dataloader in [trainer.train_dataloader, trainer.val_dataloaders]:
                if dataloader and hasattr(dataloader.dataset, 'transform'):
                    transform = dataloader.dataset.transform
                    if hasattr(transform, 'transforms'):
                        for t in transform.transforms:
                            if hasattr(t, 'size'):
                                t.size = (self.target_size, self.target_size)
                                
            self.switched = True
            print(f"✅ Resolution switch completed!")


@hydra.main(config_path='../configs', config_name='train', version_base='1.1')
def main(config: DictConfig):
    print("="*80)
    print("🚀 EfficientNet-B3 Hybrid Training - Fold 0 Only")
    print("="*80)
    print("\n📋 Configuration:")
    print(f"  • Model: EfficientNet-B3 (12.2M params)")
    print(f"  • LR: {config.models.optimizer.lr}")
    print(f"  • Weight Decay: {config.models.optimizer.weight_decay}")
    print(f"  • T_Max: {config.models.scheduler.T_max}")
    print(f"  • eta_min: {config.models.scheduler.eta_min}")
    print(f"  • Mixed Precision: FP16 ✓")
    print(f"  • Early Stopping: patience=5, monitor=val/hmean ✓")
    print(f"  • Progressive Resolution: 640px (epoch 0-3) → 960px (epoch 4+) ✓")
    print(f"  • Fold: 0 (Single fold test)")
    print("="*80 + "\n")
    
    # Set fold 0
    fold_idx = 0
    
    # Load fold data from new location
    kfold_dir = Path("/data/ephemeral/home/data/datasets/jsons/kfold")
    train_json = kfold_dir / f"fold{fold_idx}_train.json"
    val_json = kfold_dir / f"fold{fold_idx}_val.json"
    
    if not train_json.exists() or not val_json.exists():
        print(f"❌ Fold {fold_idx} data not found!")
        print(f"   Looking for: {train_json}")
        sys.exit(1)
    
    print(f"📂 Loading Fold {fold_idx} data...")
    
    # Override config with fold-specific paths
    OmegaConf.set_struct(config, False)
    config.datasets.train_dataset.annotation_path = str(train_json)
    config.datasets.val_dataset.annotation_path = str(val_json)
    
    # Update experiment name
    config.exp_name = f"{config.exp_name}_fold0"
    
    print(f"  • Train: {train_json}")
    print(f"  • Val: {val_json}\n")
    
    # Instantiate datasets
    print("🔧 Initializing datasets...")
    datasets = {
        'train': hydra.utils.instantiate(config.datasets.train_dataset),
        'val': hydra.utils.instantiate(config.datasets.val_dataset)
    }
    
    # Dataloaders
    collate_fn = DBCollateFN()
    
    train_loader = DataLoader(
        datasets['train'],
        batch_size=config.trainer.get('batch_size', 8),
        shuffle=True,
        num_workers=config.trainer.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        datasets['val'],
        batch_size=config.trainer.get('batch_size', 8),
        shuffle=False,
        num_workers=config.trainer.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Instantiate model
    print("🔧 Building EfficientNet-B3 model...")
    model = hydra.utils.instantiate(config.models)
    
    # Lightning module
    pl_module = OCRPLModule(
        model=model,
        dataset=datasets,
        config=config
    )
    
    # Callbacks
    callbacks = [
        # Model checkpoint - save best based on val/hmean
        ModelCheckpoint(
            dirpath=Path(config.checkpoint_dir) / f'fold_{fold_idx}',
            filename='best-{epoch:02d}-{val/hmean:.4f}',
            monitor='val/hmean',
            mode='max',
            save_top_k=1,
            save_last=True,
            verbose=True
        ),
        
        # Early stopping - patience 5
        EarlyStopping(
            monitor='val/hmean',
            patience=5,
            mode='max',
            verbose=True,
            min_delta=0.0001  # 0.01% minimum improvement
        ),
        
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch'),
        
        # Progressive resolution callback
        ProgressiveResolutionCallback(
            switch_epoch=4,
            initial_size=640,
            target_size=960
        )
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=f'fold_{fold_idx}'
    )
    
    # Trainer with Mixed Precision
    print("🔧 Configuring PyTorch Lightning Trainer...")
    print("  • Mixed Precision: FP16 ✓")
    print("  • Progressive Resolution: 640→960 ✓")
    print("  • Early Stopping: patience=5 ✓\n")
    
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # Mixed Precision Training (FP16)
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,  # Validate every epoch
        gradient_clip_val=1.0,
        deterministic=False,  # Faster training
        benchmark=True  # cudnn benchmark for speed
    )
    
    # Train
    print("="*80)
    print("🎯 Starting Training - Fold 0")
    print("="*80)
    trainer.fit(pl_module, train_loader, val_loader)
    
    # Best metrics
    print("\n" + "="*80)
    print("✅ Training Completed!")
    print("="*80)
    print(f"📊 Best Validation H-Mean: {trainer.callback_metrics.get('val/hmean', 0):.4f}")
    print(f"📁 Checkpoints saved to: {Path(config.checkpoint_dir) / f'fold_{fold_idx}'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
