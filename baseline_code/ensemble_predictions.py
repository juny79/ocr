#!/usr/bin/env python3
"""
Ensemble Multiple Model Predictions
Combines predictions from multiple models using weighted voting and NMS
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from shapely.geometry import Polygon
from typing import List, Dict, Tuple


def calculate_iou(box1: List, box2: List) -> float:
    """Calculate IoU between two polygons"""
    try:
        poly1 = Polygon(box1)
        poly2 = Polygon(box2)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except:
        return 0.0


def load_predictions(json_path: str) -> Dict:
    """Load prediction JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def merge_boxes(boxes: List[List], weights: List[float]) -> List:
    """Merge similar boxes using weighted average"""
    if len(boxes) == 1:
        return boxes[0]
    
    # Ensure all boxes are in numpy array format
    boxes_np = []
    for box in boxes:
        try:
            box_array = np.array(box, dtype=np.float32)
            if box_array.ndim == 1:
                # Flatten to 2D if needed
                box_array = box_array.reshape(-1, 2)
            boxes_np.append(box_array)
        except:
            # Skip malformed boxes
            continue
    
    if len(boxes_np) == 0:
        return boxes[0]
    
    if len(boxes_np) == 1:
        return boxes_np[0].tolist()
    
    # Find minimum number of points
    min_points = min(len(box) for box in boxes_np)
    max_points = max(len(box) for box in boxes_np)
    
    # Use target as average
    target_num_points = (min_points + max_points) // 2
    
    normalized_boxes = []
    valid_weights = []
    
    for i, box in enumerate(boxes_np):
        if len(box) == target_num_points:
            normalized_boxes.append(box)
            valid_weights.append(weights[i])
        elif len(box) > target_num_points:
            # Downsample
            indices = np.linspace(0, len(box) - 1, target_num_points).astype(int)
            normalized_boxes.append(box[indices])
            valid_weights.append(weights[i])
        elif len(box) >= 3:  # At least 3 points for a polygon
            # Repeat points to match target
            repeat_factor = target_num_points // len(box)
            remainder = target_num_points % len(box)
            repeated = np.repeat(box, repeat_factor, axis=0)
            if remainder > 0:
                repeated = np.vstack([repeated, box[:remainder]])
            normalized_boxes.append(repeated[:target_num_points])
            valid_weights.append(weights[i])
    
    if len(normalized_boxes) == 0:
        return boxes[0]
    
    if len(normalized_boxes) == 1:
        return normalized_boxes[0].tolist()
    
    # Stack and weight
    boxes_stacked = np.stack(normalized_boxes, axis=0)  # (N, num_points, 2)
    weights_array = np.array(valid_weights).reshape(-1, 1, 1)  # (N, 1, 1)
    
    # Weighted average
    weighted_sum = (boxes_stacked * weights_array).sum(axis=0)
    weight_total = weights_array.sum()
    
    merged = weighted_sum / weight_total
    
    return merged.tolist()


def nms_ensemble(predictions_list: List[Dict], 
                 model_weights: List[float],
                 iou_threshold: float = 0.5,
                 score_threshold: float = 0.3) -> Dict:
    """
    Ensemble predictions using NMS
    
    Args:
        predictions_list: List of prediction dicts from each model
        model_weights: Weight for each model
        iou_threshold: IoU threshold for merging boxes
        score_threshold: Minimum confidence score
    """
    
    ensemble_result = {'images': {}}
    
    # Process each image
    all_images = set()
    for pred in predictions_list:
        all_images.update(pred['images'].keys())
    
    for img_name in sorted(all_images):
        # Collect all boxes from all models for this image
        all_boxes = []
        
        for model_idx, pred in enumerate(predictions_list):
            if img_name not in pred['images']:
                continue
            
            words = pred['images'][img_name].get('words', {})
            for word_id, word_data in words.items():
                points = word_data.get('points', [])
                if len(points) > 0:
                    all_boxes.append({
                        'points': points,
                        'model_idx': model_idx,
                        'weight': model_weights[model_idx]
                    })
        
        if not all_boxes:
            ensemble_result['images'][img_name] = {'words': {}}
            continue
        
        # Cluster similar boxes
        used = [False] * len(all_boxes)
        clusters = []
        
        for i in range(len(all_boxes)):
            if used[i]:
                continue
            
            cluster = [i]
            used[i] = True
            
            for j in range(i + 1, len(all_boxes)):
                if used[j]:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(
                    all_boxes[i]['points'],
                    all_boxes[j]['points']
                )
                
                if iou > iou_threshold:
                    cluster.append(j)
                    used[j] = True
            
            clusters.append(cluster)
        
        # Merge clusters
        merged_boxes = {}
        box_idx = 1
        
        for cluster in clusters:
            # Skip if only one low-weight detection
            if len(cluster) == 1:
                box = all_boxes[cluster[0]]
                if box['weight'] < score_threshold:
                    continue
            
            # Get boxes and weights in this cluster
            cluster_boxes = [all_boxes[idx]['points'] for idx in cluster]
            cluster_weights = [all_boxes[idx]['weight'] for idx in cluster]
            
            # Merge using weighted average
            merged_points = merge_boxes(cluster_boxes, cluster_weights)
            
            merged_boxes[f'{box_idx:04d}'] = {'points': merged_points}
            box_idx += 1
        
        ensemble_result['images'][img_name] = {'words': merged_boxes}
    
    return ensemble_result


def main():
    parser = argparse.ArgumentParser(description='Ensemble model predictions')
    parser.add_argument('--model1', type=str, required=True, help='First model JSON')
    parser.add_argument('--model2', type=str, required=True, help='Second model JSON')
    parser.add_argument('--model3', type=str, required=True, help='Third model JSON')
    parser.add_argument('--weights', type=str, default='0.6,0.25,0.15', 
                        help='Model weights (comma-separated)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for merging')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path')
    
    args = parser.parse_args()
    
    # Parse weights
    weights = [float(w) for w in args.weights.split(',')]
    assert len(weights) == 3, "Must provide 3 weights"
    assert abs(sum(weights) - 1.0) < 0.01, "Weights must sum to 1.0"
    
    print(f"Loading predictions...")
    print(f"  Model 1: {args.model1} (weight={weights[0]})")
    print(f"  Model 2: {args.model2} (weight={weights[1]})")
    print(f"  Model 3: {args.model3} (weight={weights[2]})")
    
    predictions = [
        load_predictions(args.model1),
        load_predictions(args.model2),
        load_predictions(args.model3),
    ]
    
    print(f"\nEnsembling with IoU threshold={args.iou_threshold}...")
    ensemble_result = nms_ensemble(
        predictions, 
        weights,
        iou_threshold=args.iou_threshold
    )
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ensemble_result, f, indent=2)
    
    # Statistics
    total_images = len(ensemble_result['images'])
    total_boxes = sum(
        len(img_data['words']) 
        for img_data in ensemble_result['images'].values()
    )
    
    print(f"\n✅ Ensemble complete!")
    print(f"  Total images: {total_images}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Output: {output_path}")
    
    return ensemble_result


if __name__ == "__main__":
    main()
