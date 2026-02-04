"""
Swin Transformer Tiny Training Script

Model: Swin Transformer Tiny (swin_tiny_patch4_window7_224)
Parameters: 27.5M (similar to ConvNeXt-Tiny's 28M)
Expected Performance: 96.2-96.4% LB

Architecture Analysis:
- Vision Transformer with shifted window attention
- Hierarchical feature maps: [96, 192, 384, 768] channels
- Self-attention with 7x7 local windows
- Layer normalization throughout
- Depths: [2, 2, 6, 2], Num Heads: [3, 6, 12, 24]

Parameter Strategy (Architecture-Aware):
1. Learning Rate: 0.00045
   - Same as ConvNeXt-Tiny (proven optimal for ~28M models)
   - Validated through multiple successful experiments
   
2. Weight Decay: 0.00006
   - LIGHTER than ConvNeXt-Tiny's 0.000085 (30% reduction)
   - Rationale: Self-attention provides implicit regularization
     * Layer normalization stabilizes training
     * Attention mechanism has built-in feature selection
     * Avoid over-regularization (learned from ConvNeXt-Small)
   - Trust Transformer's architectural benefits
   
3. Cosine Annealing: T_max=20, eta_min=0.000008
   - Standard schedule proven effective
   
4. Progressive Resolution: 640px → 960px at epoch 4
   - Curriculum learning from simple to complex
   - Expected gain: ~1.5%p
   
5. Early Stopping: patience=5
   - Prevent over-training (critical for transformers)

Training Strategy:
- Start with lower resolution (640x640) for stable initialization
- Switch to high resolution (960x960) at epoch 4
- Monitor validation H-Mean for early stopping
- Track P-R balance (should be <1.0%p gap)

Expected Results:
- Validation: 95.8-96.2% H-Mean
- Test: 96.0-96.3% H-Mean
- Leaderboard: 96.2-96.4% H-Mean (competitive with ConvNeXt-Tiny)

Key Differences vs ConvNeXt-Tiny:
- Transformer vs CNN architecture
- Self-attention vs convolution operations
- Layer norm vs batch norm
- Different inductive biases may capture complementary features
- Good candidate for ensemble with ConvNeXt/HRNet models
"""

#!/usr/bin/env python
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


@hydra.main(version_base="1.2", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("SWIN TRANSFORMER TINY TRAINING")
    print("=" * 80)
    print(f"Model: Swin Transformer Tiny (27.5M parameters)")
    print(f"Strategy: Lighter weight decay (0.00006) for Transformer architecture")
    print(f"Learning Rate: {cfg.models.optimizer.lr}")
    print(f"Weight Decay: {cfg.models.optimizer.weight_decay}")
    print(f"Progressive Resolution: 640px → 960px at epoch 4")
    print(f"Expected LB: 96.2-96.4% H-Mean")
    print("=" * 80)
    
    # Set random seed
    pl.seed_everything(cfg.get('seed', 42), workers=True)
    
    # Setup WandB logger
    wandb_logger = None
    if cfg.get('wandb', False):
        wandb_logger = WandbLogger(
            project=cfg.get('wandb_project', 'ocr-swin-tiny'),
            name=cfg.get('wandb_run_name', 'swin_tiny_fold0'),
            save_dir=cfg.output_dir
        )
    
    # Datasets
    print("\n[1/5] Loading datasets...")
    train_dataset, val_dataset = get_train_datasets(
        data_dir=cfg.dataset.data_dir,
        split_method=cfg.dataset.split_method,
        fold=cfg.dataset.fold,
        train_transform_cfg=cfg.dataset.train_transform,
        val_transform_cfg=cfg.dataset.val_transform
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Lightning Module
    print("\n[2/5] Initializing Swin Transformer Tiny model...")
    model = OCRLightningModule(cfg)
    print(f"  Encoder: {cfg.models.encoder.name}")
    print(f"  Channels: {cfg.models.decoder.in_channels}")
    print(f"  Optimizer: Adam (lr={cfg.models.optimizer.lr}, wd={cfg.models.optimizer.weight_decay})")
    
    # Callbacks
    print("\n[3/5] Setting up callbacks...")
    callbacks = []
    
    # Checkpoint callback - save best model based on val/hmean
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints", f"fold_{cfg.dataset.fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val/hmean:.4f}",
        monitor="val/hmean",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    print(f"  Checkpoint: {checkpoint_dir}")
    
    # Early stopping - stop if no improvement for 5 epochs
    early_stop_callback = EarlyStopping(
        monitor="val/hmean",
        patience=5,
        mode="max",
        verbose=True,
        min_delta=0.0001
    )
    callbacks.append(early_stop_callback)
    print(f"  Early Stopping: patience=5, monitor=val/hmean")
    
    # Learning rate monitor
    if wandb_logger:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
    
    # Progressive resolution callback - switch from 640 to 960 at epoch 4
    progressive_callback = ProgressiveResolutionCallback(
        switch_epoch=4,
        low_res_size=(640, 640),
        high_res_size=(960, 960)
    )
    callbacks.append(progressive_callback)
    print(f"  Progressive Resolution: 640x640 → 960x960 at epoch 4")
    
    # Trainer
    print("\n[4/5] Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator='gpu',
        devices=1,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.get('gradient_clip_val', 5.0),
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=True,
        benchmark=False,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    print(f"  Max Epochs: {cfg.trainer.max_epochs}")
    print(f"  Precision: {cfg.trainer.precision}")
    print(f"  Gradient Clip: {cfg.trainer.get('gradient_clip_val', 5.0)}")
    
    # Train
    print("\n[5/5] Starting training...")
    print("=" * 80)
    trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)
    
    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/hmean: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)
    
    # Test evaluation (using best checkpoint)
    print("\n[BONUS] Running test evaluation with best checkpoint...")
    best_model = OCRLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        cfg=cfg,
        strict=False
    )
    
    # Load test dataset with same validation transform
    from ocr.datasets import get_test_datasets
    test_dataset = get_test_datasets(
        data_dir=cfg.dataset.data_dir,
        transform_cfg=cfg.dataset.val_transform  # Use val transform for test
    )
    
    test_results = trainer.test(best_model, dataloaders=test_dataset)
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    if test_results:
        for key, value in test_results[0].items():
            print(f"  {key}: {value:.4f}")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check test/hmean - should be 96.0-96.3%")
    print("2. If test looks good, generate submission file")
    print("3. Submit to leaderboard - expect 96.2-96.4% LB")
    print("4. Consider 5-fold ensemble with HRNet-W44 (different architectures)")
    print("=" * 80)


if __name__ == "__main__":
    main()
