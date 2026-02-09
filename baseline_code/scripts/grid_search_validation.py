#!/usr/bin/env python3
"""
Grid search for post-processing parameters using validation data.
This version uses validation data where we have ground truth labels.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytorch_lightning as pl

from ocr.datasets import OCRDataset
from ocr.datasets.db_collate_fn import DBCollateFN
from ocr.lightning_modules.ocr_pl import OCRPLModule
from ocr.metrics import CLEvalMetric


def load_model_and_config(checkpoint_path, config_dir):
    """Load model and configuration."""
    print(f"Loading configuration from {config_dir}")
    
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name="train", overrides=["preset=hrnet_w44_1024"])
    
    print(f"Loading model from {checkpoint_path}")
    model = OCRPLModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()
    
    return model, config


def create_val_dataloader(config):
    """Create validation dataloader."""
    # Get validation dataset config
    val_config = config.datasets.val_dataset
    
    # Create dataset using instantiate (Hydra way)
    from hydra.utils import instantiate
    val_dataset = instantiate(val_config)
    
    # Create collate function
    collate_fn = DBCollateFN()
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return val_loader


def evaluate_with_params(model, val_loader, thresh, box_thresh):
    """
    Evaluate model on validation data with given post-processing parameters.
    """
    # Update model's post-processing parameters
    model.model.head.postprocessor.thresh = thresh
    model.model.head.postprocessor.box_thresh = box_thresh
    
    # Initialize metric
    metric = CLEvalMetric()
    
    # Run inference
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].cuda()
            gt_boxes = batch['gt_boxes']
            gt_texts = batch['gt_texts']
            image_names = batch['image_name']
            
            # Forward pass
            outputs = model.model(images)
            
            # Post-process to get predictions
            pred_boxes_list, pred_scores_list = model.model.head.postprocessor(
                outputs, batch['shape']
            )
            
            # Update metric
            for pred_boxes, pred_scores, gt_box, gt_text, img_name in zip(
                pred_boxes_list, pred_scores_list, gt_boxes, gt_texts, image_names
            ):
                # Convert predictions to required format
                pred_polys = []
                for box in pred_boxes:
                    # box shape: [4, 2] -> [x1, y1, x2, y2, x3, y3, x4, y4]
                    poly = box.reshape(-1).tolist()
                    pred_polys.append(poly)
                
                # Convert ground truth to required format
                gt_polys = []
                gt_ignore = []
                for box, text in zip(gt_box, gt_text):
                    poly = box.reshape(-1).tolist()
                    gt_polys.append(poly)
                    gt_ignore.append(text == '###')  # Ignore don't care regions
                
                # Update metric
                metric.update(
                    pred_polys=pred_polys,
                    gt_polys=gt_polys,
                    gt_ignore=gt_ignore
                )
    
    # Compute final metrics
    results = metric.compute()
    
    return {
        'precision': results['precision'],
        'recall': results['recall'],
        'hmean': results['hmean']
    }


def grid_search():
    """Run grid search for post-processing parameters."""
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    CHECKPOINT_PATH = PROJECT_ROOT / "outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt"
    CONFIG_DIR = PROJECT_ROOT / "configs"
    RESULTS_DIR = PROJECT_ROOT / "grid_search_results"
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Grid search parameters
    THRESH_RANGE = np.arange(0.220, 0.246, 0.005)  # 0.220 to 0.245
    BOX_THRESH_RANGE = np.arange(0.410, 0.446, 0.005)  # 0.410 to 0.445
    
    # Baseline parameters
    BASELINE_THRESH = 0.231
    BASELINE_BOX_THRESH = 0.432
    
    print("=" * 80)
    print("Grid Search for Post-processing Parameters")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Thresh range: {THRESH_RANGE[0]:.3f} to {THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Box_thresh range: {BOX_THRESH_RANGE[0]:.3f} to {BOX_THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Total combinations: {len(THRESH_RANGE) * len(BOX_THRESH_RANGE)}")
    print(f"Baseline: thresh={BASELINE_THRESH}, box_thresh={BASELINE_BOX_THRESH}")
    print("=" * 80)
    
    # Load model and config
    print("\n[1/3] Loading model and configuration...")
    model, config = load_model_and_config(CHECKPOINT_PATH, CONFIG_DIR)
    print("✓ Model loaded successfully")
    
    # Create validation dataloader
    print("\n[2/3] Creating validation dataloader...")
    val_loader = create_val_dataloader(config)
    print(f"✓ Validation dataset loaded: {len(val_loader.dataset)} images")
    
    # Evaluate baseline
    print("\n[3/3] Evaluating baseline parameters...")
    baseline_metrics = evaluate_with_params(
        model, val_loader, BASELINE_THRESH, BASELINE_BOX_THRESH
    )
    print(f"✓ Baseline Results:")
    print(f"  - Precision: {baseline_metrics['precision']:.4f}")
    print(f"  - Recall: {baseline_metrics['recall']:.4f}")
    print(f"  - H-Mean: {baseline_metrics['hmean']:.4f}")
    
    # Run grid search
    print("\n" + "=" * 80)
    print("Starting Grid Search...")
    print("=" * 80 + "\n")
    
    results = {
        'baseline': {
            'thresh': BASELINE_THRESH,
            'box_thresh': BASELINE_BOX_THRESH,
            'metrics': baseline_metrics
        },
        'experiments': [],
        'search_space': {
            'thresh_range': THRESH_RANGE.tolist(),
            'box_thresh_range': BOX_THRESH_RANGE.tolist()
        }
    }
    
    best_hmean = baseline_metrics['hmean']
    best_params = {'thresh': BASELINE_THRESH, 'box_thresh': BASELINE_BOX_THRESH}
    
    total_experiments = len(THRESH_RANGE) * len(BOX_THRESH_RANGE)
    pbar = tqdm(total=total_experiments, desc="Grid Search Progress")
    
    for thresh in THRESH_RANGE:
        for box_thresh in BOX_THRESH_RANGE:
            try:
                # Evaluate with current parameters
                metrics = evaluate_with_params(model, val_loader, thresh, box_thresh)
                
                experiment_result = {
                    'thresh': float(thresh),
                    'box_thresh': float(box_thresh),
                    'metrics': metrics,
                    'success': True
                }
                
                # Update best
                if metrics['hmean'] > best_hmean:
                    best_hmean = metrics['hmean']
                    best_params = {'thresh': float(thresh), 'box_thresh': float(box_thresh)}
                    pbar.set_postfix({
                        'best_hmean': f"{best_hmean:.4f}",
                        'thresh': f"{thresh:.3f}",
                        'box_thresh': f"{box_thresh:.3f}"
                    })
                
            except Exception as e:
                experiment_result = {
                    'thresh': float(thresh),
                    'box_thresh': float(box_thresh),
                    'metrics': {'precision': 0.0, 'recall': 0.0, 'hmean': 0.0},
                    'success': False,
                    'error': str(e)
                }
            
            results['experiments'].append(experiment_result)
            pbar.update(1)
    
    pbar.close()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"grid_search_validation_{timestamp}.json"
    
    results['best'] = {
        'params': best_params,
        'hmean': best_hmean
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Grid Search Complete!")
    print("=" * 80)
    print(f"\nBaseline H-Mean: {baseline_metrics['hmean']:.4f}")
    print(f"Best H-Mean: {best_hmean:.4f}")
    print(f"Best Parameters:")
    print(f"  - thresh: {best_params['thresh']:.3f}")
    print(f"  - box_thresh: {best_params['box_thresh']:.3f}")
    
    improvement = best_hmean - baseline_metrics['hmean']
    print(f"\nImprovement: {improvement:+.4f} ({improvement/baseline_metrics['hmean']*100:+.2f}%)")
    
    print(f"\nResults saved to: {results_file}")
    
    # Show top 5 results
    print("\n" + "=" * 80)
    print("Top 5 Parameter Combinations:")
    print("=" * 80)
    
    sorted_results = sorted(
        [r for r in results['experiments'] if r['success']], 
        key=lambda x: x['metrics']['hmean'], 
        reverse=True
    )[:5]
    
    for i, result in enumerate(sorted_results, 1):
        print(f"\n{i}. thresh={result['thresh']:.3f}, box_thresh={result['box_thresh']:.3f}")
        print(f"   Precision: {result['metrics']['precision']:.4f}")
        print(f"   Recall: {result['metrics']['recall']:.4f}")
        print(f"   H-Mean: {result['metrics']['hmean']:.4f}")
    
    return results


if __name__ == "__main__":
    grid_search()
