#!/usr/bin/env python3
"""
Soft Voting Ensemble for 2-Fold predictions
Balances Precision and Recall by using confidence scores
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes (4 points format)"""
    def polygon_to_bbox(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [min(xs), min(ys), max(xs), max(ys)]
    
    bbox1 = polygon_to_bbox(box1)
    bbox2 = polygon_to_bbox(box2)
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def average_boxes(boxes_list, weights=None):
    """Average box coordinates with optional weights"""
    if weights is None:
        weights = [1.0] * len(boxes_list)
    
    total_weight = sum(weights)
    
    # Average all 4 points
    avg_points = []
    for point_idx in range(4):
        x = sum(box[point_idx][0] * w for box, w in zip(boxes_list, weights)) / total_weight
        y = sum(box[point_idx][1] * w for box, w in zip(boxes_list, weights)) / total_weight
        avg_points.append([round(x, 1), round(y, 1)])
    
    return avg_points


def soft_voting_ensemble(fold_predictions, iou_threshold=0.5, single_conf_threshold=0.80):
    """
    Soft voting ensemble with confidence-based filtering
    
    Args:
        fold_predictions: List of dicts with prediction data from each fold
        iou_threshold: IoU threshold for matching boxes
        single_conf_threshold: Confidence threshold for single-fold boxes (0-1)
    
    Strategy:
        - Boxes detected by both folds: Always include (high confidence)
        - Boxes detected by one fold only: Include if score > threshold
    """
    ensemble_result = {"images": {}}
    
    # Get all image names
    all_images = set()
    for fold_pred in fold_predictions:
        all_images.update(fold_pred['images'].keys())
    
    for img_name in tqdm(sorted(all_images), desc="Soft Voting Ensemble"):
        fold_boxes_list = []
        
        # Collect boxes from all folds for this image
        for fold_pred in fold_predictions:
            if img_name in fold_pred['images']:
                boxes = []
                words = fold_pred['images'][img_name]['words']
                for word_id, word_data in words.items():
                    boxes.append({
                        'points': word_data['points'],
                        'score': word_data.get('score', 0.9)  # Default score if missing
                    })
                fold_boxes_list.append(boxes)
            else:
                fold_boxes_list.append([])
        
        # Match boxes across folds
        ensemble_boxes = []
        used_indices = [set() for _ in fold_boxes_list]
        
        # Step 1: Find boxes matched across multiple folds (high confidence)
        if len(fold_boxes_list) >= 2:
            fold0_boxes = fold_boxes_list[0]
            fold1_boxes = fold_boxes_list[1]
            
            for i, box0 in enumerate(fold0_boxes):
                if i in used_indices[0]:
                    continue
                
                best_match_idx = -1
                best_iou = iou_threshold
                
                for j, box1 in enumerate(fold1_boxes):
                    if j in used_indices[1]:
                        continue
                    
                    iou = calculate_iou(box0['points'], box1['points'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = j
                
                if best_match_idx >= 0:
                    # Both folds agree - high confidence, always include
                    box1 = fold1_boxes[best_match_idx]
                    
                    # Average coordinates weighted by scores
                    weights = [box0['score'], box1['score']]
                    avg_points = average_boxes(
                        [box0['points'], box1['points']],
                        weights=weights
                    )
                    
                    ensemble_boxes.append({
                        'points': avg_points,
                        'score': (box0['score'] + box1['score']) / 2,
                        'votes': 2
                    })
                    
                    used_indices[0].add(i)
                    used_indices[1].add(best_match_idx)
        
        # Step 2: Add single-fold boxes with high confidence
        for fold_idx, boxes in enumerate(fold_boxes_list):
            for box_idx, box in enumerate(boxes):
                if box_idx not in used_indices[fold_idx]:
                    # Only one fold detected this box
                    if box['score'] >= single_conf_threshold:
                        ensemble_boxes.append({
                            'points': box['points'],
                            'score': box['score'] * 0.9,  # Slightly reduce confidence
                            'votes': 1
                        })
        
        # Convert to output format
        words_dict = {}
        for idx, box in enumerate(ensemble_boxes):
            words_dict[str(idx)] = {
                'points': box['points']
            }
        
        ensemble_result['images'][img_name] = {
            'words': words_dict
        }
    
    return ensemble_result


def main():
    parser = argparse.ArgumentParser(description='Soft Voting Ensemble')
    parser.add_argument('--fold0', type=str, required=True, help='Fold 0 prediction JSON')
    parser.add_argument('--fold1', type=str, required=True, help='Fold 1 prediction JSON')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--single-conf', type=float, default=0.80, 
                        help='Confidence threshold for single-fold boxes (0-1, default 0.80)')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading: {args.fold0}")
    with open(args.fold0, 'r') as f:
        fold0_data = json.load(f)
    
    print(f"Loading: {args.fold1}")
    with open(args.fold1, 'r') as f:
        fold1_data = json.load(f)
    
    print(f"\nSoft Voting Ensemble:")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Single-fold confidence threshold: {args.single_conf}")
    
    # Perform ensemble
    result = soft_voting_ensemble(
        [fold0_data, fold1_data],
        iou_threshold=args.iou,
        single_conf_threshold=args.single_conf
    )
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Statistics
    total_boxes = sum(len(img['words']) for img in result['images'].values())
    avg_boxes = total_boxes / len(result['images'])
    
    print(f"\n{'='*60}")
    print(f"Soft Voting Ensemble Complete!")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print(f"Total images: {len(result['images'])}")
    print(f"Total boxes: {total_boxes:,}")
    print(f"Average boxes per image: {avg_boxes:.1f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
