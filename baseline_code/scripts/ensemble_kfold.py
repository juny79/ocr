#!/usr/bin/env python
"""
K-Fold Ensemble Prediction Script
Combines predictions from all 5 folds for final submission
"""

import json
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
from tqdm import tqdm


def load_prediction(prediction_file):
    """Load prediction JSON file"""
    with open(prediction_file, 'r') as f:
        data = json.load(f)
    return data


def ensemble_predictions(pred_files, method='voting'):
    """
    Ensemble multiple prediction files
    
    Args:
        pred_files: List of prediction JSON file paths
        method: 'voting' or 'average'
    
    Returns:
        Ensembled predictions
    """
    print(f"Ensembling {len(pred_files)} predictions using {method}...")
    
    # Load all predictions
    all_preds = [load_prediction(f) for f in pred_files]
    
    # Get all image filenames
    image_filenames = list(all_preds[0]['images'].keys())
    
    ensembled = OrderedDict(images=OrderedDict())
    
    for img_name in tqdm(image_filenames, desc="Ensembling"):
        # Collect all boxes from all folds for this image
        all_boxes = []
        
        for pred in all_preds:
            if img_name in pred['images']:
                words = pred['images'][img_name]['words']
                for word_id, word_data in words.items():
                    all_boxes.append(word_data['points'])
        
        if method == 'voting':
            # NMS-based voting: remove duplicates and keep confident boxes
            final_boxes = nms_boxes(all_boxes, iou_threshold=0.5)
        else:
            # Simple approach: use first fold's prediction
            final_boxes = all_boxes[:len(all_preds[0]['images'][img_name]['words'])]
        
        # Create words dict
        words = OrderedDict()
        for idx, box in enumerate(final_boxes):
            words[f'{idx + 1:04}'] = OrderedDict(points=box)
        
        ensembled['images'][img_name] = OrderedDict(words=words)
    
    return ensembled


def nms_boxes(boxes, iou_threshold=0.5):
    """
    Non-Maximum Suppression for boxes
    Simple version: keep boxes that appear in multiple folds
    """
    if len(boxes) == 0:
        return []
    
    # Count occurrences of similar boxes
    box_scores = defaultdict(int)
    unique_boxes = []
    
    for box in boxes:
        # Find if similar box already exists
        found = False
        for idx, unique_box in enumerate(unique_boxes):
            if boxes_iou(box, unique_box) > iou_threshold:
                box_scores[idx] += 1
                found = True
                break
        
        if not found:
            unique_boxes.append(box)
            box_scores[len(unique_boxes) - 1] = 1
    
    # Sort by score (how many folds predicted this box)
    sorted_indices = sorted(box_scores.keys(), key=lambda x: box_scores[x], reverse=True)
    
    # FILTERING: Keep boxes with at least 2 votes (40% agreement)
    # 2 votes was good (0.9176), 3 votes gave low recall (0.68).
    
    final_indices = [idx for idx in sorted_indices if box_scores[idx] >= 2]
    
    print(f"  Removed {len(sorted_indices) - len(final_indices)} low-confidence boxes (votes < 2)")
    
    # Return boxes sorted by confidence
    return [unique_boxes[idx] for idx in final_indices]


def boxes_iou(box1, box2):
    """Calculate IoU between two boxes"""
    try:
        # Convert to numpy arrays
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        # Get bounding boxes
        x1_min, y1_min = box1.min(axis=0)
        x1_max, y1_max = box1.max(axis=0)
        x2_min, y2_min = box2.min(axis=0)
        x2_max, y2_max = box2.max(axis=0)
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    except:
        return 0.0


def save_prediction(data, output_file):
    """Save prediction to JSON file"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to: {output_file}")


def main():
    """Main ensemble function"""
    print("="*70)
    print("K-FOLD ENSEMBLE PREDICTION")
    print("="*70)
    
    # Find all prediction files
    predictions_dir = Path('/data/ephemeral/home/baseline_code/predictions/kfold')
    pred_files = sorted(predictions_dir.glob('fold*_predictions.json'))
    
    if len(pred_files) == 0:
        print("❌ No prediction files found!")
        print("   Please run predictions for each fold first.")
        return
    
    print(f"\nFound {len(pred_files)} prediction files:")
    for pf in pred_files:
        print(f"  - {pf.name}")
    
    # Ensemble predictions
    ensembled = ensemble_predictions(pred_files, method='voting')
    
    # Save ensembled prediction
    output_file = '/data/ephemeral/home/baseline_code/predictions/kfold_ensemble.json'
    save_prediction(ensembled, output_file)
    
    print(f"\n✓ K-Fold ensemble complete!")
    print(f"  Output: {output_file}")
    print(f"  Total images: {len(ensembled['images'])}")


if __name__ == '__main__':
    main()
