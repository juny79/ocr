#!/usr/bin/env python3
"""
HRNet-W44 1280 5-Fold Training with WandB Sweep Optimal Parameters
===================================================================
Architecture: HRNet-W44 (57M parameters)
Resolution: 1280x1280
Optimal Parameters (from WandB Sweep mfwh1uoz):
- lr: 0.0001 (9.978e-05)
- weight_decay: 0.00004 (3.980e-05)
- T_max: 15
- eta_min: 0.00001 (1.112e-05)

Expected Performance: 97.56% val/hmean (validated in sweep)

K-Fold Configuration:
- Checkpoint: save_top_k=3 (only top 3 checkpoints per fold)
- Early stopping: patience=5
- Total time estimate: ~10-12 hours (5 folds)

Usage:
    cd /data/ephemeral/home/baseline_code
    source /data/ephemeral/home/.env
    python runners/train_hrnet_w44_1280_optimal_kfold.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
import hydra
from omegaconf import OmegaConf
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
from ocr.datasets.kfold_split import KFoldSplitter
from ocr.datasets import get_datasets_by_cfg

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train_optimal_kfold(config):
    """
    Train HRNet-W44 1280 with WandB Sweep optimal parameters across 5 folds.
    """
    
    # Apply optimal parameters from sweep
    print("\n" + "="*80)
    print("HRNet-W44 1280 K-Fold Training with Optimal Parameters")
    print("="*80)
    print("\nOptimal Parameters (WandB Sweep mfwh1uoz):")
    print(f"  - Learning Rate: 0.0001")
    print(f"  - Weight Decay: 0.00004")
    print(f"  - T_max: 15")
    print(f"  - eta_min: 0.00001")
    print(f"  - Expected val/hmean: 0.97561")
    print("="*80 + "\n")
    
    # Override config with optimal parameters
    OmegaConf.set_struct(config, False)  # Allow adding new keys
    config.models.optimizer.lr = 0.0001
    config.models.optimizer.weight_decay = 0.00004
    config.models.scheduler.T_max = 15
    config.models.scheduler.eta_min = 0.00001
    config.trainer.max_epochs = 20  # Allow up to 20 epochs (early stopping will handle)
    OmegaConf.set_struct(config, True)  # Lock config again
    
    # Initialize K-Fold splitter
    kfold_splitter = KFoldSplitter(n_splits=5, random_state=42)
    
    # Get training dataset to create fold splits
    datasets = get_datasets_by_cfg(config.datasets)
    train_dataset = datasets['train']
    
    # Create fold mapping
    fold_mapping_dir = Path('kfold_results')
    fold_mapping_dir.mkdir(exist_ok=True)
    fold_mapping_file = fold_mapping_dir / 'fold_mapping.json'
    
    if fold_mapping_file.exists():
        print(f"Loading existing fold mapping from {fold_mapping_file}")
        with open(fold_mapping_file, 'r') as f:
            fold_mapping = json.load(f)
    else:
        print(f"Creating new fold mapping...")
        fold_mapping = kfold_splitter.save_fold_mapping(
            train_dataset.anns,
            fold_mapping_file
        )
    
    fold_results = {}
    
    for fold_idx in range(5):
        print(f"\n{'='*80}")
        print(f"Training Fold {fold_idx}/4 - HRNet-W44 1280")
        print(f"{'='*80}")
        print(f"Fold {fold_idx} - Train: {len(fold_mapping[f'fold_{fold_idx}']['train'])} samples")
        print(f"Fold {fold_idx} - Val: {len(fold_mapping[f'fold_{fold_idx}']['val'])} samples")
        print(f"{'='*80}\n")
        
        # Set seed for reproducibility
        pl.seed_everything(config.get("seed", 42) + fold_idx, workers=True)
        
        # Update config for this fold
        fold_train_keys = fold_mapping[f"fold_{fold_idx}"]["train"]
        fold_val_keys = fold_mapping[f"fold_{fold_idx}"]["val"]
        
        # Create fold-specific datasets using KFoldDataset
        from ocr.datasets.kfold_split import KFoldDataset
        
        # Get base datasets
        datasets = get_datasets_by_cfg(config.datasets)
        
        # Wrap train and val datasets for this fold
        datasets['train'] = KFoldDataset(
            datasets['train'], 
            fold_idx=fold_idx, 
            split='train',
            kfold_splitter=kfold_splitter
        )
        datasets['val'] = KFoldDataset(
            datasets['val'],
            fold_idx=fold_idx,
            split='val',
            kfold_splitter=kfold_splitter
        )
        
        # Get model and data modules
        model_module, data_module = get_pl_modules_by_cfg(config)
        
        # Update data module datasets
        data_module.dataset = datasets
        model_module.dataset = datasets
        
        # Logger setup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        exp_name = f"hrnet_w44_1280_optimal_fold{fold_idx}"
        
        if config.get("wandb", False):
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(
                entity='fc_bootcamp',  # Team entity
                project='hrnet-w44-1280-optimal-kfold',
                name=f"HRNet-W44-1280-Optimal Fold {fold_idx}",
                config={
                    'fold': fold_idx,
                    'lr': 0.0001,
                    'weight_decay': 0.00004,
                    'T_max': 15,
                    'eta_min': 0.00001,
                    'architecture': 'HRNet-W44',
                    'resolution': '1280x1280',
                    'source': 'WandB Sweep mfwh1uoz',
                },
                tags=["hrnet-w44", "1280", "optimal", "kfold", f"fold-{fold_idx}"],
            )
        else:
            logger = True
        
        # Callbacks with save_top_k=3
        checkpoint_dir = f"outputs/hrnet_w44_1280_optimal_fold{fold_idx}/checkpoints"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="epoch={epoch}-step={step}",
                monitor="val/hmean",
                mode="max",
                save_top_k=3,  # Only keep top 3 checkpoints
                save_last=True,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(
                monitor="val/hmean",
                mode="max",
                patience=5,
                verbose=True,
                min_delta=0.0001,
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
            precision="32",  # Use FP32 to avoid autocast issues with BCE Loss
            enable_progress_bar=True,
            gradient_clip_val=1.0,
        )
        
        print(f"\n🚀 Starting training for Fold {fold_idx}...")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Checkpoints: Saving top 3 based on val/hmean\n")
        
        # Train
        try:
            trainer.fit(model_module, data_module)
            
            # Get best checkpoint
            best_ckpt = trainer.checkpoint_callback.best_model_path
            print(f"\n✅ Fold {fold_idx} Training Complete!")
            print(f"Best checkpoint: {best_ckpt}")
            print(f"Best val/hmean: {trainer.checkpoint_callback.best_model_score:.6f}")
            
            # Test with best checkpoint
            if best_ckpt:
                test_results = trainer.test(
                    model_module,
                    datamodule=data_module,
                    ckpt_path=best_ckpt
                )
                
                if test_results:
                    fold_results[f"fold_{fold_idx}"] = {
                        'test_results': test_results[0],
                        'best_checkpoint': best_ckpt,
                        'best_score': float(trainer.checkpoint_callback.best_model_score),
                    }
            
        except Exception as e:
            print(f"\n❌ Error in Fold {fold_idx}: {str(e)}")
            fold_results[f"fold_{fold_idx}"] = {'error': str(e)}
            continue
        
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx} Complete - Moving to next fold")
        print(f"{'='*80}\n")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*80)
    print("HRNet-W44 1280 Optimal K-Fold Training Complete!")
    print("="*80)
    print(f"\nFold Results Summary:")
    
    scores = []
    for fold_idx in range(5):
        fold_key = f"fold_{fold_idx}"
        if fold_key in fold_results:
            result = fold_results[fold_key]
            if 'error' in result:
                print(f"  Fold {fold_idx}: ERROR - {result['error']}")
            else:
                score = result.get('best_score', 0)
                scores.append(score)
                print(f"  Fold {fold_idx}: {score:.6f} (hmean)")
                print(f"    Checkpoint: {result.get('best_checkpoint', 'N/A')}")
    
    # Calculate statistics
    if scores:
        avg_score = sum(scores) / len(scores)
        std_score = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
        print(f"\n📊 Performance Statistics:")
        print(f"  Average: {avg_score:.6f} ± {std_score:.6f}")
        print(f"  Best: {max(scores):.6f}")
        print(f"  Worst: {min(scores):.6f}")
        print(f"  Range: {max(scores) - min(scores):.6f}")
    
    print("\n" + "="*80 + "\n")
    
    # Save results
    results_file = f"hrnet_w44_1280_optimal_kfold_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'optimal_params': {
                'lr': 0.0001,
                'weight_decay': 0.00004,
                'T_max': 15,
                'eta_min': 0.00001,
            },
            'fold_results': fold_results,
            'summary': {
                'avg_score': avg_score if scores else None,
                'std_score': std_score if scores else None,
                'best_score': max(scores) if scores else None,
                'worst_score': min(scores) if scores else None,
            }
        }, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}\n")
    print("Next steps:")
    print("  1. Review checkpoint files in outputs/hrnet_w44_1280_optimal_fold*/checkpoints/")
    print("  2. Use best checkpoints for ensemble prediction")
    print("  3. Expected ensemble performance: 97.5-97.7% hmean\n")


if __name__ == "__main__":
    train_optimal_kfold()
