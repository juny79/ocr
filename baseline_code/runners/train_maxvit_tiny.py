#!/usr/bin/env python3
"""
MaxViT-Tiny Hybrid Training Script
====================================
Architecture: Multi-axis Attention + Convolution (Hybrid)
Parameters: 28.6M (similar to ConvNeXt-Tiny)

Strategy:
- Uses proven parameters from ConvNeXt-Tiny (same param count)
- lr: 0.00045, weight_decay: 0.000085
- Progressive resolution: 640 -> 960
- Expected LB: 96.1-96.3%

Usage:
    python runners/train_maxvit_tiny.py preset=maxvit_tiny_hybrid
"""

import os
import sys
from datetime import datetime
import lightning.pytorch as pl
import hydra
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train(config):
    """Train MaxViT-Tiny model with proven parameters."""
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    # Logging setup
    if config.get("wandb"):
        from lightning.pytorch.loggers import WandbLogger as Logger
        import wandb
        
        exp_name = f"maxvit_tiny_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        logger = Logger(
            project=config.get('wandb_project', 'ocr-maxvit-tiny-experiment'),
            name=config.get('wandb_run_name', exp_name),
            config=dict(config),
            log_model=True,
            tags=["maxvit-tiny", "hybrid-attention", "progressive-resolution"],
        )
    else:
        logger = True

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=f"./checkpoints/maxvit_tiny_{datetime.now().strftime('%Y%m%d_%H%M')}",
            filename="best_model",
            monitor="val_metric",
            mode="max",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_metric",
            mode="max",
            patience=5,
            verbose=True,
        ),
    ]

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        precision="16-mixed",
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model_module, data_module)

    # Test
    trainer.test(model_module, data_module)

    print("\n" + "="*80)
    print("MaxViT-Tiny Training Complete!")
    print("="*80)
    print(f"Model: MaxViT-Tiny (28.6M parameters)")
    print(f"Architecture: Multi-axis Attention + Convolution (Hybrid)")
    print(f"Resolution Strategy: Progressive (640 -> 960)")
    print(f"LR: 0.00045, Weight Decay: 0.000085")
    print(f"Expected LB: 96.1-96.3%")
    print("="*80 + "\n")


if __name__ == "__main__":
    train()
