#!/usr/bin/env python3
"""
5-Fold Ensemble with multiple voting strategies
Supports both hard voting and soft voting
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


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


def ensemble_5fold(fold_predictions, voting_threshold=3, iou_threshold=0.5):
    """
    5-Fold ensemble with voting
    
    Args:
        fold_predictions: List of 5 prediction dicts (one per fold)
        voting_threshold: Minimum number of folds that must agree (1-5)
        iou_threshold: IoU threshold for matching boxes
    
    Returns:
        Ensemble prediction dict
    """
    ensemble_result = {"images": {}}
    
    # Get all image names
    all_images = set()
    for fold_pred in fold_predictions:
        all_images.update(fold_pred['images'].keys())
    
    stats = defaultdict(int)
    
    for img_name in tqdm(sorted(all_images), desc=f"5-Fold Ensemble (voting≥{voting_threshold})"):
        # Collect boxes from all folds for this image
        fold_boxes_list = []
        for fold_pred in fold_predictions:
            if img_name in fold_pred['images']:
                boxes = []
                words = fold_pred['images'][img_name]['words']
                for word_id, word_data in words.items():
                    boxes.append({
                        'points': word_data['points'],
                        'fold_id': fold_predictions.index(fold_pred)
                    })
                fold_boxes_list.append(boxes)
            else:
                fold_boxes_list.append([])
        
        # Match boxes across folds using IoU
        all_boxes = []
        for fold_idx, boxes in enumerate(fold_boxes_list):
            for box in boxes:
                all_boxes.append(box)
        
        # Group overlapping boxes
        box_groups = []
        used_indices = set()
        
        for i, box1 in enumerate(all_boxes):
            if i in used_indices:
                continue
            
            group = [box1]
            used_indices.add(i)
            
            for j, box2 in enumerate(all_boxes):
                if j in used_indices:
                    continue
                
                # Check if box2 overlaps with any box in group
                for box_in_group in group:
                    iou = calculate_iou(box_in_group['points'], box2['points'])
                    if iou > iou_threshold:
                        group.append(box2)
                        used_indices.add(j)
                        break
        
            box_groups.append(group)
        
        # Filter by voting threshold and average coordinates
        ensemble_boxes = []
        for group in box_groups:
            vote_count = len(group)
            
            if vote_count >= voting_threshold:
                # Average box coordinates
                box_points = [box['points'] for box in group]
                avg_points = average_boxes(box_points)
                
                ensemble_boxes.append({
                    'points': avg_points,
                    'votes': vote_count
                })
                
                stats[f'votes_{vote_count}'] += 1
        
        # Convert to output format
        words_dict = {}
        for idx, box in enumerate(ensemble_boxes):
            words_dict[str(idx)] = {
                'points': box['points']
            }
        
        ensemble_result['images'][img_name] = {
            'words': words_dict
        }
    
    return ensemble_result, stats


def main():
    parser = argparse.ArgumentParser(description='5-Fold Ensemble')
    parser.add_argument('--fold0', type=str, required=True, help='Fold 0 prediction JSON')
    parser.add_argument('--fold1', type=str, required=True, help='Fold 1 prediction JSON')
    parser.add_argument('--fold2', type=str, required=True, help='Fold 2 prediction JSON')
    parser.add_argument('--fold3', type=str, required=True, help='Fold 3 prediction JSON')
    parser.add_argument('--fold4', type=str, required=True, help='Fold 4 prediction JSON')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--voting', type=int, default=3, 
                        help='Voting threshold (1-5, default 3)')
    parser.add_argument('--iou', type=float, default=0.5, 
                        help='IoU threshold for matching (default 0.5)')
    
    args = parser.parse_args()
    
    # Validate voting threshold
    if args.voting < 1 or args.voting > 5:
        raise ValueError("Voting threshold must be between 1 and 5")
    
    # Load predictions
    fold_files = [args.fold0, args.fold1, args.fold2, args.fold3, args.fold4]
    fold_predictions = []
    
    print(f"{'='*70}")
    print(f"Loading 5-Fold Predictions")
    print(f"{'='*70}")
    
    for idx, fold_file in enumerate(fold_files):
        print(f"Fold {idx}: {fold_file}")
        with open(fold_file, 'r') as f:
            fold_predictions.append(json.load(f))
    
    print(f"\n5-Fold Ensemble Configuration:")
    print(f"  Voting threshold: {args.voting} (out of 5 folds)")
    print(f"  IoU threshold: {args.iou}")
    print(f"{'='*70}\n")
    
    # Perform ensemble
    result, stats = ensemble_5fold(
        fold_predictions,
        voting_threshold=args.voting,
        iou_threshold=args.iou
    )
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Statistics
    total_boxes = sum(len(img['words']) for img in result['images'].values())
    avg_boxes = total_boxes / len(result['images'])
    
    print(f"\n{'='*70}")
    print(f"5-Fold Ensemble Complete!")
    print(f"{'='*70}")
    print(f"Output: {args.output}")
    print(f"Total images: {len(result['images'])}")
    print(f"Total boxes: {total_boxes:,}")
    print(f"Average boxes per image: {avg_boxes:.1f}")
    print(f"\nVoting distribution:")
    for votes in sorted([k for k in stats.keys() if k.startswith('votes_')], 
                        key=lambda x: int(x.split('_')[1])):
        count = stats[votes]
        vote_num = int(votes.split('_')[1])
        print(f"  {vote_num}/5 folds agreed: {count:,} boxes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
