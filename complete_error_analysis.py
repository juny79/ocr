"""
Complete Error Case Analysis with Model Inference
Fold 3 최고 성능 모델로 Validation Set 예측 및 에러 분석
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add baseline code to path
sys.path.append('/data/ephemeral/home/baseline_code')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    from ocr.lightning_modules.ocr_pl import OCRModule
    
    print(f"Loading model from {checkpoint_path}...")
    model = OCRModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
    print("Model loaded successfully!")
    return model


def load_val_data(val_json_path):
    """Load validation ground truth"""
    with open(val_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['images']


def predict_image(model, img_path, max_size=1024):
    """Run inference on single image"""
    from ocr.datasets.transforms import get_transforms
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    # Apply transform
    transform = get_transforms(is_train=False, max_size=max_size)
    transformed = transform(image=img)
    img_tensor = transformed['image'].unsqueeze(0)
    
    # Move to device
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Post-process to get boxes
    boxes = model.post_process(outputs, orig_w, orig_h)
    
    return boxes


def polygon_to_array(points):
    """Convert points to numpy array"""
    return np.array(points, dtype=np.float32)


def calculate_iou(box1, box2):
    """Calculate IoU between two polygons"""
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    
    try:
        poly1 = Polygon(box1)
        poly2 = Polygon(box2)
        
        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)
            
        if poly1.is_empty or poly2.is_empty:
            return 0.0
            
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predicted boxes with ground truth boxes"""
    matched = []
    false_positives = []
    matched_gt = set()
    
    # For each prediction, find best matching GT
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            
            gt_box = gt_boxes[gt_idx]
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
        else:
            false_positives.append((pred_idx, best_iou))
    
    # Find false negatives (GT boxes not matched)
    false_negatives = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    
    return {
        'matched': matched,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def analyze_image(img_name, pred_boxes, gt_info):
    """Analyze single image performance"""
    gt_boxes = [polygon_to_array(word['points']) for word in gt_info['words'].values()]
    
    match_result = match_boxes(pred_boxes, gt_boxes)
    
    tp = len(match_result['matched'])
    fp = len(match_result['false_positives'])
    fn = len(match_result['false_negatives'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'image_name': img_name,
        'gt_count': len(gt_boxes),
        'pred_count': len(pred_boxes),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'match_result': match_result,
        'gt_boxes': gt_boxes,
        'pred_boxes': pred_boxes
    }


def visualize_error(result, img_path, output_path, rank):
    """Visualize error case"""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Ground Truth
    axes[0].imshow(img)
    axes[0].set_title(f'Ground Truth ({result["gt_count"]} boxes)', 
                     fontsize=14, fontweight='bold')
    for box in result['gt_boxes']:
        poly = patches.Polygon(box, linewidth=2, edgecolor='green', 
                             facecolor='none', alpha=0.8)
        axes[0].add_patch(poly)
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(img)
    axes[1].set_title(f'Predictions ({result["pred_count"]} boxes)', 
                     fontsize=14, fontweight='bold')
    for box in result['pred_boxes']:
        poly = patches.Polygon(box, linewidth=2, edgecolor='blue', 
                             facecolor='none', alpha=0.8)
        axes[1].add_patch(poly)
    axes[1].axis('off')
    
    # Error Overlay
    axes[2].imshow(img)
    title = f'F1: {result["f1_score"]:.3f} | P: {result["precision"]:.3f} | R: {result["recall"]:.3f}'
    axes[2].set_title(title, fontsize=14, fontweight='bold')
    
    match_result = result['match_result']
    
    # True Positives (green)
    for pred_idx, gt_idx, iou in match_result['matched']:
        poly = patches.Polygon(result['pred_boxes'][pred_idx], 
                             linewidth=2, edgecolor='green', 
                             facecolor='none', alpha=0.7)
        axes[2].add_patch(poly)
    
    # False Positives (red)
    for pred_idx, _ in match_result['false_positives']:
        poly = patches.Polygon(result['pred_boxes'][pred_idx], 
                             linewidth=3, edgecolor='red', 
                             facecolor='none')
        axes[2].add_patch(poly)
    
    # False Negatives (yellow)
    for gt_idx in match_result['false_negatives']:
        poly = patches.Polygon(result['gt_boxes'][gt_idx], 
                             linewidth=3, edgecolor='yellow', 
                             facecolor='none')
        axes[2].add_patch(poly)
    
    axes[2].axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=3, 
              label=f'TP: {result["true_positives"]}'),
        Line2D([0], [0], color='red', lw=3, 
              label=f'FP: {result["false_positives"]}'),
        Line2D([0], [0], color='yellow', lw=3, 
              label=f'FN: {result["false_negatives"]}')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.suptitle(f'Error Case #{rank:02d}: {result["image_name"]}', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Configuration
    CHECKPOINT = '/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt'
    VAL_JSON = '/data/ephemeral/home/data/datasets/jsons/val.json'
    IMG_DIR = Path('/data/ephemeral/home/data/datasets/images')
    OUTPUT_DIR = Path('/data/ephemeral/home/error_analysis')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("ERROR CASE ANALYSIS - FOLD 3 BEST MODEL")
    print("="*80)
    print()
    
    # Load ground truth
    print("Loading validation ground truth...")
    gt_data = load_val_data(VAL_JSON)
    print(f"  - Loaded {len(gt_data)} validation images")
    print()
    
    # Load model
    try:
        model = load_model(CHECKPOINT)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nFallback: Using pre-computed predictions if available")
        print("Or run inference separately and provide predictions")
        return
    
    # Run inference on validation set
    print("Running inference on validation set...")
    print(f"  (Processing {len(gt_data)} images...)")
    
    all_results = []
    
    for img_name in tqdm(list(gt_data.keys())[:50]):  # Process first 50 for speed
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            continue
        
        # Predict
        pred_boxes = predict_image(model, img_path)
        
        # Analyze
        result = analyze_image(img_name, pred_boxes, gt_data[img_name])
        all_results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'image_name': r['image_name'],
        'gt_count': r['gt_count'],
        'pred_count': r['pred_count'],
        'true_positives': r['true_positives'],
        'false_positives': r['false_positives'],
        'false_negatives': r['false_negatives'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1_score': r['f1_score']
    } for r in all_results])
    
    # Save results
    df.to_csv(OUTPUT_DIR / 'per_image_performance.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'per_image_performance.csv'}")
    
    # Overall metrics
    total_tp = df['true_positives'].sum()
    total_fp = df['false_positives'].sum()
    total_fn = df['false_negatives'].sum()
    
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    print("\n" + "="*80)
    print("OVERALL VALIDATION PERFORMANCE")
    print("="*80)
    print(f"Precision: {overall_p:.4f}")
    print(f"Recall:    {overall_r:.4f}")
    print(f"F1-Score:  {overall_f1:.4f}")
    print()
    
    # Find error cases
    error_cases = df[df['f1_score'] < 1.0].sort_values('f1_score')
    
    print(f"Error cases (F1 < 1.0): {len(error_cases)}/{len(df)}")
    print(f"\nWorst 10 cases:")
    print(error_cases[['image_name', 'f1_score', 'precision', 'recall', 
                       'false_positives', 'false_negatives']].head(10).to_string(index=False))  
    print()
    
    # Visualize top error cases
    print("Visualizing top 20 error cases...")
    for rank, (idx, row) in enumerate(error_cases.head(20).iterrows(), 1):
        result = all_results[idx]
        img_path = IMG_DIR / result['image_name']
        output_path = OUTPUT_DIR / f'error_{rank:02d}_f1{result["f1_score"]:.3f}_{result["image_name"]}.png'
        visualize_error(result, img_path, output_path, rank)
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    try:
        import shapely
        main()
    except ImportError:
        print("Error: Required library not installed")
        print("Please install: pip install shapely")
