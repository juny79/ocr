#!/usr/bin/env python3
"""
WBF (Weighted Boxes Fusion) Ensemble
- 박스를 merge하지 않고 모두 유지
- Confidence-weighted averaging으로 겹치는 박스만 조정
"""

import json
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
import argparse
from typing import List, Dict


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


def wbf_ensemble(predictions_list: List[Dict], 
                 model_weights: List[float],
                 iou_threshold: float = 0.5,
                 skip_box_threshold: float = 0.0) -> Dict:
    """
    WBF Ensemble: Keep all boxes, only fuse overlapping ones
    
    Key difference from NMS:
    - All boxes are kept (not deleted)
    - Only overlapping boxes are fused with confidence weighting
    - Final confidence = average of model confidences for fused boxes
    """
    
    ensemble_result = {'images': {}}
    
    # Process each image
    all_images = set()
    for pred in predictions_list:
        all_images.update(pred['images'].keys())
    
    for img_name in sorted(all_images):
        # Collect all boxes from all models
        all_boxes = []
        
        for model_idx, pred in enumerate(predictions_list):
            if img_name not in pred['images']:
                continue
            
            words = pred['images'][img_name].get('words', {})
            for word_id, word_data in words.items():
                points = word_data.get('points', [])
                if len(points) >= 3:
                    all_boxes.append({
                        'points': points,
                        'confidence': model_weights[model_idx],
                        'model_idx': model_idx,
                        'original_id': word_id
                    })
        
        if not all_boxes:
            ensemble_result['images'][img_name] = {'words': {}}
            continue
        
        # Sort by confidence (descending)
        all_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        # WBF clustering
        clusters = []
        used = [False] * len(all_boxes)
        
        for i in range(len(all_boxes)):
            if used[i]:
                continue
            
            # Start new cluster with this box
            cluster = {
                'boxes': [all_boxes[i]],
                'indices': [i],
                'total_conf': all_boxes[i]['confidence']
            }
            used[i] = True
            
            # Find overlapping boxes
            for j in range(i + 1, len(all_boxes)):
                if used[j]:
                    continue
                
                # Check IoU with cluster representative (first box)
                iou = calculate_iou(
                    all_boxes[i]['points'],
                    all_boxes[j]['points']
                )
                
                if iou >= iou_threshold:
                    cluster['boxes'].append(all_boxes[j])
                    cluster['indices'].append(j)
                    cluster['total_conf'] += all_boxes[j]['confidence']
                    used[j] = True
            
            clusters.append(cluster)
        
        # Generate output boxes
        output_boxes = {}
        box_idx = 1
        
        for cluster in clusters:
            # Skip low confidence clusters
            avg_conf = cluster['total_conf'] / len(cluster['boxes'])
            if avg_conf < skip_box_threshold:
                continue
            
            # Fuse boxes in cluster using confidence weighting
            if len(cluster['boxes']) == 1:
                # Single box - use as-is
                fused_points = cluster['boxes'][0]['points']
            else:
                # Multiple boxes - weighted fusion
                fused_points = fuse_boxes_weighted(
                    [b['points'] for b in cluster['boxes']],
                    [b['confidence'] for b in cluster['boxes']]
                )
            
            output_boxes[f'{box_idx:04d}'] = {'points': fused_points}
            box_idx += 1
        
        ensemble_result['images'][img_name] = {'words': output_boxes}
    
    return ensemble_result


def fuse_boxes_weighted(boxes: List[List], confidences: List[float]) -> List:
    """
    Fuse multiple boxes using confidence-weighted averaging
    Handles different polygon sizes by finding common representation
    """
    if len(boxes) == 1:
        return boxes[0]
    
    # Normalize confidences
    conf_sum = sum(confidences)
    weights = [c / conf_sum for c in confidences]
    
    # Convert to arrays and handle different point counts
    max_points = max(len(box) for box in boxes)
    min_points = min(len(box) for box in boxes)
    
    # Use median point count as target
    target_points = sorted([len(box) for box in boxes])[len(boxes) // 2]
    
    normalized_boxes = []
    for box, weight in zip(boxes, weights):
        box_array = np.array(box, dtype=np.float32)
        
        if len(box) == target_points:
            normalized_boxes.append((box_array, weight))
        elif len(box) > target_points:
            # Downsample
            indices = np.linspace(0, len(box) - 1, target_points).astype(int)
            normalized_boxes.append((box_array[indices], weight))
        else:
            # Upsample by interpolation
            indices = np.linspace(0, len(box) - 1, target_points)
            interp_box = np.zeros((target_points, 2))
            for dim in range(2):
                interp_box[:, dim] = np.interp(
                    indices,
                    np.arange(len(box)),
                    box_array[:, dim]
                )
            normalized_boxes.append((interp_box, weight))
    
    # Weighted average
    fused = np.zeros((target_points, 2))
    for box_array, weight in normalized_boxes:
        fused += box_array * weight
    
    return fused.tolist()


def main():
    parser = argparse.ArgumentParser(description='WBF Ensemble')
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--model3', type=str, required=True)
    parser.add_argument('--weights', type=str, default='0.6,0.25,0.15',
                        help='Model confidence weights')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for box fusion')
    parser.add_argument('--skip-threshold', type=float, default=0.0,
                        help='Skip boxes below this confidence')
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    # Parse weights
    weights = [float(w) for w in args.weights.split(',')]
    assert len(weights) == 3
    
    print(f"WBF Ensemble Configuration:")
    print(f"  Model 1: {args.model1} (conf={weights[0]})")
    print(f"  Model 2: {args.model2} (conf={weights[1]})")
    print(f"  Model 3: {args.model3} (conf={weights[2]})")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Skip threshold: {args.skip_threshold}")
    
    # Load predictions
    predictions = []
    for model_path in [args.model1, args.model2, args.model3]:
        with open(model_path, 'r') as f:
            predictions.append(json.load(f))
    
    print(f"\nRunning WBF ensemble...")
    ensemble_result = wbf_ensemble(
        predictions,
        weights,
        iou_threshold=args.iou_threshold,
        skip_box_threshold=args.skip_threshold
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ensemble_result, f)
    
    # Statistics
    total_images = len(ensemble_result['images'])
    total_boxes = sum(
        len(img_data['words']) 
        for img_data in ensemble_result['images'].values()
    )
    
    print(f"\n✅ WBF Ensemble complete!")
    print(f"  Total images: {total_images}")
    print(f"  Total boxes: {total_boxes:,}")
    print(f"  Avg boxes/image: {total_boxes/total_images:.1f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
