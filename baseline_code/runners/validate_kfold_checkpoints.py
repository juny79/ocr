#!/usr/bin/env python
"""
Direct K-Fold Prediction - Load checkpoints directly without Hydra
"""

import sys
import torch
from pathlib import Path

sys.path.append('/data/ephemeral/home/baseline_code')

# Import after adding to path
import lightning.pytorch as pl
from ocr.lightning_modules.ocr_pl import OCRLitModule

# Fold checkpoints
FOLD_CHECKPOINTS = {
    'fold_0': 'outputs/hrnet_w44_1280_optimal_fold0/checkpoints/epoch=4-step=6545.ckpt',
    'fold_1': 'outputs/hrnet_w44_1280_optimal_fold1/checkpoints/epoch=3-step=5236.ckpt',
    'fold_2': 'outputs/hrnet_w44_1280_optimal_fold2/checkpoints/epoch=18-step=24871.ckpt',
    'fold_3': 'outputs/hrnet_w44_1280_optimal_fold3/checkpoints/epoch=4-step=6545.ckpt',
    'fold_4': 'outputs/hrnet_w44_1280_optimal_fold4/checkpoints/epoch=17-step=23544.ckpt',
}


def test_fold(fold_name, checkpoint_path):
    """Test a single fold"""
    print(f"\n{'='*70}")
    print(f"Testing: {fold_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    base_dir = Path('/data/ephemeral/home/baseline_code')
    checkpoint_full_path = base_dir / checkpoint_path
    
    # Check checkpoint
    if not checkpoint_full_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_full_path}")
        return False
    
    try:
        # Load checkpoint directly
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_full_path, map_location='cpu')
        
        # Get hyper parameters from checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            print(f"✓ Loaded hyperparameters")
            print(f"  Model: {hparams.get('model_name', 'Unknown')}")
        else:
            print("⚠️ No hyperparameters found in checkpoint")
            return False
        
        # Load model from checkpoint
        print("Loading model...")
        model = OCRLitModule.load_from_checkpoint(
            str(checkpoint_full_path),
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {'CUDA' if next(model.parameters()).is_cuda else 'CPU'}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # TODO: Setup data module and run prediction
        # For now, just confirm checkpoint can be loaded
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution"""
    print("="*70)
    print("K-FOLD CHECKPOINT VALIDATION")
    print("="*70)
    print(f"\nValidating {len(FOLD_CHECKPOINTS)} checkpoints...")
    print()
    
    success_count = 0
    
    for fold_name, checkpoint_path in FOLD_CHECKPOINTS.items():
        if test_fold(fold_name, checkpoint_path):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"Validation Summary: {success_count}/{len(FOLD_CHECKPOINTS)} checkpoints valid")
    print(f"{'='*70}\n")
    
    if success_count == len(FOLD_CHECKPOINTS):
        print("✓ All checkpoints are valid!")
        print("\nTo generate predictions, use the training script's test mode")
        print("or implement data loading in this script.")
    else:
        print(f"❌ {len(FOLD_CHECKPOINTS) - success_count} checkpoint(s) failed")


if __name__ == '__main__':
    main()
