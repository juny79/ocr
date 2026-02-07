#!/usr/bin/env python
"""
Generate submission from Fold 4 checkpoint
"""

import os
import sys
sys.path.append('/data/ephemeral/home/baseline_code')

import torch
import lightning.pytorch as pl
from hydra import compose, initialize
from pathlib import Path

# Set environment variable for config directory
os.environ['OP_CONFIG_DIR'] = '/data/ephemeral/home/baseline_code/configs'

from ocr.lightning_modules import get_pl_modules_by_cfg

def main():
    print("="*70)
    print("FOLD 4 SUBMISSION GENERATION")
    print("="*70)
    print()
    
    # Checkpoint path
    checkpoint_path = "/data/ephemeral/home/baseline_code/outputs/hrnet_w44_1280_optimal_fold4/checkpoints/epoch=17-step=23544.ckpt"
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Val Hmean: 0.9837 (Best across all folds)")
    print()
    
    # Check checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize Hydra
    with initialize(version_base='1.2', config_path='../configs'):
        # Load config
        cfg = compose(
            config_name='predict',
            overrides=[
                'preset=hrnet_w44_1280',
            ]
        )
        
        # Override checkpoint path in config
        cfg.checkpoint_path = checkpoint_path
        
        print("✓ Config loaded")
        print()
        
        # Set seed
        pl.seed_everything(cfg.get("seed", 42), workers=True)
        
        # Create modules
        print("Creating model and data modules...")
        model_module, data_module = get_pl_modules_by_cfg(cfg)
        print("✓ Modules created")
        print()
        
        # Create trainer
        trainer = pl.Trainer(
            logger=False,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
        )
        
        # Run prediction
        print("Running prediction on test set...")
        print("This may take 10-15 minutes...")
        print()
        
        predictions = trainer.predict(
            model_module,
            data_module,
            ckpt_path=checkpoint_path
        )
        
        print()
        print("="*70)
        print("✓ PREDICTION COMPLETE!")
        print("="*70)
        print()
        print("Predictions saved to outputs directory")
        print("Check for submission CSV file")
        print()


if __name__ == '__main__':
    main()
