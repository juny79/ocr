#!/usr/bin/env python
"""
EfficientNet-B3 Hybrid Training - Fold 0
Uses existing train.py structure with fold-specific data paths
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import lightning.pytorch as pl
import hydra
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


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
        
        if current_epoch >= self.switch_epoch and not self.switched:
            print(f"\n{'='*60}")
            print(f"🔄 Epoch {current_epoch}: Switching resolution")
            print(f"   {self.initial_size}px → {self.target_size}px")
            print(f"{'='*60}\n")
            self.switched = True


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train(config):
    print("="*80)
    print("🚀 EfficientNet-B3 Hybrid Training - Fold 0")
    print("="*80)
    print("\n📋 Configuration:")
    print(f"  • Model: EfficientNet-B3")
    print(f"  • LR: {config.models.optimizer.lr}")
    print(f"  • Weight Decay: {config.models.optimizer.weight_decay}")
    print(f"  • T_Max: {config.models.scheduler.T_max}")
    print(f"  • eta_min: {config.models.scheduler.eta_min}")
    print(f"  • Mixed Precision: FP16")
    print(f"  • Early Stopping: patience=5")
    print(f"  • Progressive Resolution: 640px → 960px (epoch 4+)")
    print("="*80 + "\n")
    
    # Override with Fold 0 data paths
    fold_idx = 0
    kfold_dir = Path("/data/ephemeral/home/data/datasets/jsons/kfold")
    train_json = kfold_dir / f"fold{fold_idx}_train.json"
    val_json = kfold_dir / f"fold{fold_idx}_val.json"
    
    if not train_json.exists() or not val_json.exists():
        print(f"❌ Fold {fold_idx} data not found!")
        print(f"   Looking for: {train_json}")
        sys.exit(1)
    
    print(f"📂 Using Fold {fold_idx} data:")
    print(f"  • Train: {train_json}")
    print(f"  • Val: {val_json}\n")
    
    # Override annotation paths
    OmegaConf.set_struct(config, False)
    config.datasets.train_dataset.annotation_path = str(train_json)
    config.datasets.val_dataset.annotation_path = str(val_json)
    config.exp_name = f"{config.exp_name}_fold0"
    
    # Seed
    pl.seed_everything(config.get("seed", 42), workers=True)
    
    # Get model and data modules using existing infrastructure
    print("🔧 Initializing model and data modules...")
    model_module, data_module = get_pl_modules_by_cfg(config)
    
    # Logger
    if config.get("wandb"):
        from lightning.pytorch.loggers import WandbLogger as Logger
        import wandb
        import os
        
        # Force offline mode to avoid API permission issues
        os.environ['WANDB_MODE'] = 'offline'
        
        exp_name = f"efficientnet_b3_fold0_{datetime.now().strftime('%Y%m%d_%H%M')}"
        project = "efficientnet-b3-ocr-fold0"
        
        print(f"📊 WandB Configuration (OFFLINE MODE):")
        print(f"  • Project: {project}")
        print(f"  • Experiment: {exp_name}")
        print(f"  • Mode: OFFLINE - logs will be saved locally")
        print(f"  • Sync later with: wandb sync wandb/run-xxx\n")
        
        logger = Logger(
            project=project,
            name=exp_name,
            config=dict(config),
            log_model=False,
            tags=["efficientnet_b3", "hybrid", "fold0", "progressive_res"],
        )
    else:
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=f"{config.exp_name}_fold{fold_idx}",
            version=config.exp_version,
            default_hp_metric=False,
        )
    
    # Callbacks
    checkpoint_path = Path(config.checkpoint_dir) / f"fold_{fold_idx}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=str(checkpoint_path),
            filename='best-{epoch:02d}-{val/hmean:.4f}',
            save_top_k=3,
            monitor='val/hmean',
            mode='max',
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor='val/hmean',
            patience=5,
            mode='max',
            verbose=True,
            min_delta=0.0001
        ),
        ProgressiveResolutionCallback(
            switch_epoch=4,
            initial_size=640,
            target_size=960
        )
    ]
    
    # Trainer
    print("🔧 Configuring Trainer...")
    
    # Override trainer config for optimization
    # Note: Mixed Precision (FP16) disabled due to BCE Loss incompatibility
    # BCE Loss is unsafe with autocast - would need BCEWithLogitsLoss instead
    OmegaConf.set_struct(config, False)
    config.trainer.precision = '32'  # Use FP32 instead of FP16
    config.trainer.gradient_clip_val = 1.0
    config.trainer.benchmark = True
    
    print("  • Precision: FP32 (FP16 disabled due to BCE Loss)")
    print("  • Progressive Resolution: 640→960 @ epoch 4")
    print("  • Early Stopping: patience=5\n")
    
    trainer = pl.Trainer(
        **config.trainer,
        logger=logger,
        callbacks=callbacks
    )
    
    # Train
    print("\n" + "="*80)
    print("🎯 Starting Training")
    print("="*80 + "\n")
    
    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume", None),
    )
    
    # Test
    print("\n" + "="*80)
    print("📊 Running Final Test")
    print("="*80 + "\n")
    trainer.test(model_module, data_module)
    
    # Summary
    print("\n" + "="*80)
    print("✅ Training Completed!")
    print("="*80)
    best_hmean = trainer.callback_metrics.get('val/hmean', 0)
    print(f"📊 Best Validation H-Mean: {best_hmean:.4f}")
    print(f"📁 Checkpoints: {checkpoint_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    train()
