#!/usr/bin/env python
"""
Simple K-Fold Prediction Script
Directly loads checkpoints and generates predictions
"""

import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
import lightning.pytorch as pl
from hydra import compose, initialize_config_dir

sys.path.append('/data/ephemeral/home/baseline_code')
from ocr.lightning_modules import get_pl_modules_by_cfg

# Fold checkpoints
FOLD_CHECKPOINTS = {
    'fold_0': 'outputs/hrnet_w44_1280_optimal_fold0/checkpoints/epoch=4-step=6545.ckpt',
    'fold_1': 'outputs/hrnet_w44_1280_optimal_fold1/checkpoints/epoch=3-step=5236.ckpt',
    'fold_2': 'outputs/hrnet_w44_1280_optimal_fold2/checkpoints/epoch=18-step=24871.ckpt',
    'fold_3': 'outputs/hrnet_w44_1280_optimal_fold3/checkpoints/epoch=4-step=6545.ckpt',
    'fold_4': 'outputs/hrnet_w44_1280_optimal_fold4/checkpoints/epoch=17-step=23544.ckpt',
}


def load_config(preset='hrnet_w44_1280', checkpoint_path=''):
    """Load prediction config using Hydra"""
    config_dir = '/data/ephemeral/home/baseline_code/configs'
    
    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=config_dir, version_base='1.2'):
        # Compose config with overrides
        cfg = compose(
            config_name='predict',
            overrides=[
                f'preset={preset}',
                f'+checkpoint_path={checkpoint_path}',
            ]
        )
    
    return cfg


def run_prediction(fold_name, checkpoint_path):
    """Run prediction for a single fold"""
    print(f"\n{'='*70}")
    print(f"Predicting: {fold_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    base_dir = Path('/data/ephemeral/home/baseline_code')
    checkpoint_full_path = base_dir / checkpoint_path
    
    # Check checkpoint
    if not checkpoint_full_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_full_path}")
        return False
    
    # Load config
    config = load_config('hrnet_w44_1280', str(checkpoint_full_path))
    
    # Set seed
    pl.seed_everything(config.get("seed", 42), workers=True)
    
    # Create modules
    try:
        model_module, data_module = get_pl_modules_by_cfg(config)
    except Exception as e:
        print(f"❌ Failed to create modules: {e}")
        return False
    
    # Create trainer
    trainer = pl.Trainer(
        logger=False,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    
    # Run prediction
    try:
        print(f"Running prediction...")
        predictions = trainer.predict(
            model_module,
            data_module,
            ckpt_path=str(checkpoint_full_path)
        )
        
        print(f"✓ {fold_name} prediction complete")
        print(f"  Generated {len(predictions)} prediction batches")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution"""
    print("="*70)
    print("K-FOLD ENSEMBLE PREDICTION")
    print("="*70)
    print(f"\nProcessing {len(FOLD_CHECKPOINTS)} folds...")
    print()
    
    success_count = 0
    
    # Run predictions for each fold
    for fold_name, checkpoint_path in FOLD_CHECKPOINTS.items():
        if run_prediction(fold_name, checkpoint_path):
            success_count += 1
        else:
            print(f"\n⚠️ Warning: {fold_name} failed, continuing...")
    
    print(f"\n{'='*70}")
    print(f"Prediction Summary: {success_count}/{len(FOLD_CHECKPOINTS)} folds completed")
    print(f"{'='*70}\n")
    
    if success_count > 0:
        print("✓ Predictions generated successfully!")
        print("\nNext steps:")
        print("  1. Check outputs/hrnet_w44_1280_optimal_fold*/predictions/")
        print("  2. Run ensemble script to combine predictions")
        print("  3. Convert to submission CSV")
    else:
        print("❌ All predictions failed")


if __name__ == '__main__':
    main()
