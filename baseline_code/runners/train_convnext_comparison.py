#!/usr/bin/env python3
"""
ConvNeXt-Tiny/Small 빠른 비교 학습 스크립트
EfficientNet-B3와 동일한 하이브리드 파라미터 사용
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ocr.lightning_modules import get_pl_modules_by_cfg

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


class ProgressiveResolutionCallback(pl.Callback):
    """Progressive Resolution: 640px → 960px at epoch 4"""
    
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


def train_model(model_name: str, preset: str):
    """Train single model"""
    print(f"\n{'='*80}")
    print(f"🚀 Training {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Force offline mode for WandB
    os.environ['WANDB_MODE'] = 'offline'
    
    # Load config
    with hydra.initialize(config_path="../configs/preset", version_base=None):
        cfg = hydra.compose(config_name=preset)
    
    print(f"📋 Configuration:")
    print(f"  • Model: {model_name}")
    print(f"  • LR: {cfg.models.optimizer.lr}")
    print(f"  • Weight Decay: {cfg.models.optimizer.weight_decay}")
    print(f"  • T_max: {cfg.models.scheduler.T_max}")
    print(f"  • eta_min: {cfg.models.scheduler.eta_min}")
    print(f"  • Precision: FP32")
    print(f"  • Early Stopping: patience=5")
    print(f"  • Progressive Resolution: 640px → 960px (epoch 4+)")
    
    # Data paths
    data_root = Path("/data/ephemeral/home/data/datasets")
    train_json = data_root / "jsons/kfold/fold0_train.json"
    val_json = data_root / "jsons/kfold/fold0_val.json"
    
    print(f"\n📂 Using Fold 0 data:")
    print(f"  • Train: {train_json}")
    print(f"  • Val: {val_json}")
    
    # Datasets with initial 640px resolution
    train_transform = get_transforms(image_size=640, is_train=True)
    val_transform = get_transforms(image_size=640, is_train=False)
    
    train_dataset = DBDataset(
        json_path=str(train_json),
        image_dir=str(data_root / "images"),
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = DBDataset(
        json_path=str(val_json),
        image_dir=str(data_root / "images"),
        transform=val_transform,
        is_train=False
    )
    
    print(f"  • Train samples: {len(train_dataset)}")
    print(f"  • Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=db_collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=db_collate_fn,
        pin_memory=True
    )
    
    # Model
    model = OCRPL(cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔢 Model Parameters:")
    print(f"  • Total: {total_params:,}")
    print(f"  • Trainable: {trainable_params:,}")
    
    # Callbacks
    checkpoint_dir = project_root / f"outputs/{cfg.exp_name}_fold0/checkpoints/fold_0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-epoch={epoch:02d}-val/hmean={val/hmean:.4f}',
        monitor='val/hmean',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/hmean',
        patience=5,
        mode='max',
        min_delta=0.0001,
        verbose=True
    )
    
    progressive_callback = ProgressiveResolutionCallback(
        switch_epoch=4,
        initial_size=640,
        target_size=960
    )
    
    # WandB Logger (offline mode)
    wandb_logger = WandbLogger(
        project=f'{model_name}-ocr-fold0',
        name=f'{model_name}_hybrid_fold0',
        save_dir=str(project_root / 'wandb'),
        offline=True
    )
    
    print(f"\n📊 WandB Configuration (OFFLINE MODE):")
    print(f"  • Project: {model_name}-ocr-fold0")
    print(f"  • Mode: OFFLINE - logs saved locally")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='32',  # FP32 for BCE Loss compatibility
        callbacks=[checkpoint_callback, early_stop_callback, progressive_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    print(f"\n{'='*80}")
    print(f"🎯 Starting Training - {model_name.upper()}")
    print(f"{'='*80}\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Results
    print(f"\n{'='*80}")
    print(f"✅ Training Complete - {model_name.upper()}")
    print(f"{'='*80}")
    print(f"\nBest Checkpoint:")
    print(f"  Path: {checkpoint_callback.best_model_path}")
    print(f"  Score: {checkpoint_callback.best_model_score:.4f}")
    
    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    train_model()
