#!/usr/bin/env python3
"""
Fixed Ensemble: Ensures each cluster has at most 1 box per model
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
    """
    Merge boxes using weighted average
    Always returns 4 points (quadrilateral)
    """
    if len(boxes) == 1:
        return boxes[0]
    
    # Convert all boxes to numpy arrays
    valid_boxes = []
    valid_weights = []
    
    for box, weight in zip(boxes, weights):
        try:
            box_array = np.array(box, dtype=np.float32)
            if box_array.ndim == 1:
                box_array = box_array.reshape(-1, 2)
            
            # Skip invalid boxes
            if len(box_array) < 3:
                continue
            
            # Convert to 4 points if needed
            if len(box_array) == 4:
                valid_boxes.append(box_array)
                valid_weights.append(weight)
            else:
                # Sample/interpolate to 4 points
                if len(box_array) > 4:
                    indices = np.linspace(0, len(box_array) - 1, 4).astype(int)
                    valid_boxes.append(box_array[indices])
                else:
                    # Repeat points to get 4
                    repeated = np.tile(box_array, (4 // len(box_array) + 1, 1))[:4]
                    valid_boxes.append(repeated)
                valid_weights.append(weight)
        except:
            continue
    
    if len(valid_boxes) == 0:
        return boxes[0]
    
    if len(valid_boxes) == 1:
        return valid_boxes[0].tolist()
    
    # Stack and compute weighted average
    boxes_stacked = np.stack(valid_boxes, axis=0)  # (N, 4, 2)
    weights_array = np.array(valid_weights).reshape(-1, 1, 1)  # (N, 1, 1)
    weights_array = weights_array / weights_array.sum()  # Normalize
    
    merged = (boxes_stacked * weights_array).sum(axis=0)  # (4, 2)
    
    return merged.tolist()


def nms_ensemble(predictions_list: List[Dict], 
                 model_weights: List[float],
                 iou_threshold: float = 0.5) -> Dict:
    """
    Fixed ensemble: Each cluster contains at most 1 box per model
    """
    
    ensemble_result = {'images': {}}
    
    # Process each image
    all_images = set()
    for pred in predictions_list:
        all_images.update(pred['images'].keys())
    
    for img_name in sorted(all_images):
        # Collect boxes from each model separately
        model_boxes = []
        
        for model_idx, pred in enumerate(predictions_list):
            boxes_from_model = []
            
            if img_name in pred['images']:
                words = pred['images'][img_name].get('words', {})
                for word_id, word_data in words.items():
                    points = word_data.get('points', [])
                    if len(points) >= 3:
                        boxes_from_model.append({
                            'points': points,
                            'model_idx': model_idx,
                            'weight': model_weights[model_idx],
                            'word_id': word_id
                        })
            
            model_boxes.append(boxes_from_model)
        
        # Build clusters: each cluster has at most 1 box from each model
        used = [set() for _ in range(len(predictions_list))]
        clusters = []
        
        # Use model 0 as anchor
        for i, box_i in enumerate(model_boxes[0]):
            if i in used[0]:
                continue
            
            cluster = {0: box_i}
            used[0].add(i)
            
            # Find matching boxes from other models
            for model_idx in range(1, len(predictions_list)):
                best_match = None
                best_iou = iou_threshold
                
                for j, box_j in enumerate(model_boxes[model_idx]):
                    if j in used[model_idx]:
                        continue
                    
                    iou = calculate_iou(box_i['points'], box_j['points'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = (j, box_j)
                
                if best_match is not None:
                    j, box_j = best_match
                    cluster[model_idx] = box_j
                    used[model_idx].add(j)
            
            clusters.append(cluster)
        
        # Add remaining boxes from other models
        for model_idx in range(1, len(predictions_list)):
            for j, box_j in enumerate(model_boxes[model_idx]):
                if j in used[model_idx]:
                    continue
                
                # Check if it matches any existing cluster
                matched = False
                for cluster in clusters:
                    if model_idx in cluster:
                        continue  # Cluster already has a box from this model
                    
                    # Check IoU with any box in cluster
                    for other_box in cluster.values():
                        iou = calculate_iou(box_j['points'], other_box['points'])
                        if iou > iou_threshold:
                            cluster[model_idx] = box_j
                            used[model_idx].add(j)
                            matched = True
                            break
                    
                    if matched:
                        break
                
                # Create new cluster if no match
                if not matched:
                    clusters.append({model_idx: box_j})
                    used[model_idx].add(j)
        
        # Merge clusters
        merged_boxes = {}
        box_idx = 1
        
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            
            # Get boxes and weights
            cluster_boxes = [box['points'] for box in cluster.values()]
            cluster_weights = [box['weight'] for box in cluster.values()]
            
            # Merge
            merged_points = merge_boxes(cluster_boxes, cluster_weights)
            
            merged_boxes[f'{box_idx:04d}'] = {'points': merged_points}
            box_idx += 1
        
        ensemble_result['images'][img_name] = {'words': merged_boxes}
    
    return ensemble_result


def main():
    parser = argparse.ArgumentParser(description='Fixed ensemble')
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--model3', type=str, required=True)
    parser.add_argument('--weights', type=str, default='0.6,0.25,0.15')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    weights = [float(w) for w in args.weights.split(',')]
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 0.01
    
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
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ensemble_result, f)
    
    total_images = len(ensemble_result['images'])
    total_boxes = sum(
        len(img_data['words']) 
        for img_data in ensemble_result['images'].values()
    )
    
    print(f"\n✅ Fixed ensemble complete!")
    print(f"  Total images: {total_images}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
