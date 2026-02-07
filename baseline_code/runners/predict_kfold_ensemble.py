#!/usr/bin/env python
"""
K-Fold Ensemble Prediction Script
Runs inference on all 5 folds and creates ensemble submission
"""

import os
import sys
import json
from pathlib import Path
import subprocess
import time

# Fold checkpoints from K-Fold training results
FOLD_CHECKPOINTS = {
    'fold_0': {
        'checkpoint': 'outputs/hrnet_w44_1280_optimal_fold0/checkpoints/epoch=4-step=6545.ckpt',
        'val_hmean': 0.973811,
    },
    'fold_1': {
        'checkpoint': 'outputs/hrnet_w44_1280_optimal_fold1/checkpoints/epoch=3-step=5236.ckpt',
        'val_hmean': 0.971670,
    },
    'fold_2': {
        'checkpoint': 'outputs/hrnet_w44_1280_optimal_fold2/checkpoints/epoch=18-step=24871.ckpt',
        'val_hmean': 0.978108,
    },
    'fold_3': {
        'checkpoint': 'outputs/hrnet_w44_1280_optimal_fold3/checkpoints/epoch=4-step=6545.ckpt',
        'val_hmean': 0.976367,
    },
    'fold_4': {
        'checkpoint': 'outputs/hrnet_w44_1280_optimal_fold4/checkpoints/epoch=17-step=23544.ckpt',
        'val_hmean': 0.983676,
    },
}


def run_prediction(fold_name, checkpoint_path, output_dir):
    """Run prediction for a single fold"""
    print(f"\n{'='*70}")
    print(f"Running prediction for {fold_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create temporary config file for this fold
    import tempfile
    import yaml
    
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    config_data = {
        'defaults': [
            '_self_',
            {'preset': 'hrnet_w44_1280'},
            {'override hydra/hydra_logging': 'disabled'},
            {'override hydra/job_logging': 'disabled'},
        ],
        'seed': 42,
        'exp_name': f'{fold_name}_prediction',
        'checkpoint_path': str(checkpoint_path),
        'minified_json': False,
    }
    
    yaml.dump(config_data, temp_config)
    temp_config.close()
    
    # Run prediction command using temporary config
    cmd = [
        'python', 'runners/predict.py',
        '--config-name', temp_config.name.replace('.yaml', '').split('/')[-1],
        '--config-path', os.path.dirname(temp_config.name),
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/data/ephemeral/home/baseline_code',
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Clean up temp file
        os.unlink(temp_config.name)
        
        if result.returncode == 0:
            print(f"✓ {fold_name} prediction complete")
            return True
        else:
            print(f"❌ {fold_name} prediction failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        os.unlink(temp_config.name)
        print(f"❌ {fold_name} prediction timed out")
        return False
    except Exception as e:
        os.unlink(temp_config.name)
        print(f"❌ {fold_name} prediction error: {e}")
        return False


def ensemble_predictions(pred_dir, output_file):
    """Ensemble predictions from all folds"""
    print(f"\n{'='*70}")
    print("Creating ensemble prediction")
    print(f"{'='*70}\n")
    
    # Run ensemble script
    cmd = [
        'python', 'scripts/ensemble_kfold.py',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/data/ephemeral/home/baseline_code',
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0:
            print("✓ Ensemble prediction complete")
            print(result.stdout)
            return True
        else:
            print("❌ Ensemble prediction failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Ensemble prediction error: {e}")
        return False


def convert_to_submission(json_file, csv_file):
    """Convert JSON prediction to CSV submission"""
    print(f"\n{'='*70}")
    print("Converting to submission format")
    print(f"{'='*70}\n")
    
    cmd = [
        'python', 'ocr/utils/convert_submission.py',
        '--input', json_file,
        '--output', csv_file,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/data/ephemeral/home/baseline_code',
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Submission CSV created: {csv_file}")
            return True
        else:
            print("❌ Conversion failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False


def main():
    """Main execution"""
    print("="*70)
    print("K-FOLD ENSEMBLE PREDICTION PIPELINE")
    print("="*70)
    print(f"\nFolds to process: {len(FOLD_CHECKPOINTS)}")
    print(f"Expected ensemble performance: 97.5-97.7% hmean")
    print()
    
    # Base directories
    base_dir = Path('/data/ephemeral/home/baseline_code')
    pred_base_dir = base_dir / 'predictions' / 'kfold'
    pred_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Run predictions for each fold
    success_count = 0
    for fold_name, fold_info in FOLD_CHECKPOINTS.items():
        checkpoint_path = base_dir / fold_info['checkpoint']
        output_dir = pred_base_dir / fold_name
        
        if run_prediction(fold_name, checkpoint_path, output_dir):
            success_count += 1
            time.sleep(2)  # Brief pause between folds
        else:
            print(f"\n⚠️ Warning: {fold_name} prediction failed, continuing...")
    
    print(f"\n{'='*70}")
    print(f"Prediction Summary: {success_count}/{len(FOLD_CHECKPOINTS)} folds completed")
    print(f"{'='*70}\n")
    
    if success_count == 0:
        print("❌ No predictions succeeded. Aborting.")
        return
    
    # Step 2: Ensemble predictions
    ensemble_json = base_dir / 'predictions' / 'kfold_ensemble.json'
    if not ensemble_predictions(pred_base_dir, ensemble_json):
        print("❌ Ensemble failed. Aborting.")
        return
    
    # Step 3: Convert to submission CSV
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    submission_csv = base_dir / f'submissions/kfold_ensemble_{timestamp}.csv'
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    
    if convert_to_submission(ensemble_json, submission_csv):
        print(f"\n{'='*70}")
        print("✓ K-FOLD ENSEMBLE PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📊 Results:")
        print(f"  - Folds processed: {success_count}/{len(FOLD_CHECKPOINTS)}")
        print(f"  - Ensemble JSON: {ensemble_json}")
        print(f"  - Submission CSV: {submission_csv}")
        print(f"\n💡 Next step:")
        print(f"  Submit {submission_csv.name} to competition")
        print()
    else:
        print("❌ Conversion to CSV failed")


if __name__ == '__main__':
    main()
