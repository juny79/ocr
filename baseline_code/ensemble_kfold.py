#!/usr/bin/env python3
"""
5-Fold K-Fold Ensemble Script
평균 기반 앙상블 (동일 아키텍처, 동일 파라미터 모델)
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def calculate_iou_polygon(poly1, poly2):
    """Calculate IoU between two polygons"""
    from shapely.geometry import Polygon
    from shapely import validation
    
    try:
        # Handle both formats: [[x,y], [x,y], ...] and [x, y, x, y, ...]
        if isinstance(poly1[0], list):
            coords1 = [(pt[0], pt[1]) for pt in poly1]
        else:
            coords1 = [(poly1[i], poly1[i+1]) for i in range(0, len(poly1), 2)]
            
        if isinstance(poly2[0], list):
            coords2 = [(pt[0], pt[1]) for pt in poly2]
        else:
            coords2 = [(poly2[i], poly2[i+1]) for i in range(0, len(poly2), 2)]
        
        p1 = Polygon(coords1)
        p2 = Polygon(coords2)
        
        # Make valid if needed
        if not p1.is_valid:
            p1 = validation.make_valid(p1)
        if not p2.is_valid:
            p2 = validation.make_valid(p2)
        
        intersection = p1.intersection(p2).area
        union = p1.union(p2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except Exception as e:
        # print(f"IoU calculation error: {e}")  # Debug
        return 0.0


def ensemble_kfold_predictions(json_files, iou_threshold=0.75, weights=None, min_vote=3):
    """
    5-Fold 예측 결과를 앙상블
    
    Args:
        json_files: 5개 fold의 JSON 파일 경로 리스트
        iou_threshold: 동일 박스로 간주할 IoU threshold (높게 설정)
        weights: 각 fold의 가중치 (None이면 성능 기반 가중치 사용)
        min_vote: 최소 투표 수 (기본값: 3, 즉 5개 중 3개 이상 fold가 동의해야 함)
    
    Returns:
        앙상블된 예측 결과 딕셔너리
    """
    n_folds = len(json_files)
    
    if weights is None:
        # 성능 기반 가중치 (Leaderboard 및 Test 성능 기준)
        # Fold 3: 0.9863 (best), Fold 0: 0.9851, Fold 2: 0.9830, Fold 1: 0.9829, Fold 4: 0.9816
        weights = [0.25, 0.15, 0.15, 0.30, 0.15]  # Fold 0, 1, 2, 3, 4
        print("Using performance-based weights:")
        print("  Fold 0 (0.9851): 0.25")
        print("  Fold 1 (0.9829): 0.15")
        print("  Fold 2 (0.9830): 0.15")
        print("  Fold 3 (0.9863): 0.30")
        print("  Fold 4 (0.9816): 0.15")
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    print(f"\nEnsemble Configuration:")
    print(f"  Folds: {n_folds}")
    print(f"  IoU Threshold: {iou_threshold}")
    print(f"  Min Vote: {min_vote}/{n_folds}")
    print(f"  Weights: {weights}")
    
    # Load all predictions
    all_predictions = []
    for i, json_file in enumerate(json_files):
        print(f"Loading Fold {i}: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            images_data = data['images']
            all_predictions.append(images_data)
            n_images = len(images_data)
            n_boxes = sum(len(img_data['words']) for img_data in images_data.values())
            print(f"  Images: {n_images}, Total boxes: {n_boxes}")
    
    # Get all image names
    image_names = set()
    for predictions in all_predictions:
        image_names.update(predictions.keys())
    
    print(f"\nEnsembling {len(image_names)} images...")
    
    # Ensemble by image
    ensemble_result = {}
    debug_count = 0
    
    for img_name in tqdm(sorted(image_names)):
        # Collect predictions for this image from all folds
        fold_predictions = []
        for predictions in all_predictions:
            if img_name in predictions:
                # Convert dict of words to list of boxes
                words = predictions[img_name]['words']
                boxes = [{'points': word['points']} for word in words.values()]
                fold_predictions.append(boxes)
            else:
                fold_predictions.append([])
        
        # Ensemble boxes
        ensemble_boxes = ensemble_boxes_for_image(fold_predictions, weights, iou_threshold, min_vote, debug=(debug_count < 3))
        debug_count += 1
        
        ensemble_result[img_name] = {
            'words': {f'{i:04d}': box for i, box in enumerate(ensemble_boxes, start=1)}
        }
    
    # Statistics
    total_boxes = sum(len(img_data['words']) for img_data in ensemble_result.values())
    avg_boxes = total_boxes / len(ensemble_result)
    
    print(f"\nEnsemble Results:")
    print(f"  Total images: {len(ensemble_result)}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Avg boxes per image: {avg_boxes:.1f}")
    
    return ensemble_result


def ensemble_boxes_for_image(fold_predictions, weights, iou_threshold, min_vote, debug=False):
    """
    하나의 이미지에 대한 박스들을 앙상블
    
    Strategy:
    1. 모든 fold의 박스를 수집
    2. 겹치는 박스들을 클러스터링 (높은 IoU threshold)
    3. 각 클러스터가 min_vote 이상의 fold에서 나왔는지 확인
    4. 유효한 클러스터 내에서 가중 평균 (성능 기반 가중치)
    """
    # Collect all boxes from all folds
    all_boxes = []
    for fold_idx, boxes in enumerate(fold_predictions):
        for box in boxes:
            all_boxes.append({
                'points': box['points'],
                'fold_idx': fold_idx,
                'weight': weights[fold_idx]
            })
    
    if not all_boxes:
        return []
    
    # Cluster boxes based on IoU
    clusters = []
    used = set()
    
    for i, box_i in enumerate(all_boxes):
        if i in used:
            continue
        
        # Start new cluster
        cluster = [box_i]
        cluster_folds = {box_i['fold_idx']}
        used.add(i)
        
        # Find overlapping boxes (same box from different folds)
        for j in range(i + 1, len(all_boxes)):
            if j in used:
                continue
            
            box_j = all_boxes[j]
            
            # Skip if this fold is already in the cluster
            if box_j['fold_idx'] in cluster_folds:
                continue
            
            # Check IoU with any box in the cluster
            max_iou = 0
            for box_in_cluster in cluster:
                iou = calculate_iou_polygon(box_j['points'], box_in_cluster['points'])
                max_iou = max(max_iou, iou)
            
            # Add to cluster if IoU is high enough
            if max_iou > iou_threshold:
                cluster.append(box_j)
                cluster_folds.add(box_j['fold_idx'])
                used.add(j)
        
        clusters.append(cluster)
    
    # Merge each cluster (only if min_vote satisfied)
    ensemble_boxes = []
    
    for cluster in clusters:
        # Count unique folds in this cluster
        unique_folds = len(set(box['fold_idx'] for box in cluster))
        
        # Only keep boxes that appear in at least min_vote folds
        if unique_folds >= min_vote:
            # Use coordinates from best fold (Fold 3) instead of averaging
            # This preserves the exact coordinates that achieved 0.9863
            best_box = select_best_box_from_cluster(cluster)
            ensemble_boxes.append(best_box)
    
    # Debug: print stats
    if debug:
        fold_counts = [len(pred) for pred in fold_predictions]
        cluster_sizes = [len(c) for c in clusters]
        cluster_votes = [len(set(box['fold_idx'] for box in c)) for c in clusters]
        print(f"\nDEBUG:")
        print(f"  Fold box counts: {fold_counts}, Total: {sum(fold_counts)}")
        print(f"  Clusters formed: {len(clusters)}")
        print(f"  Cluster sizes (first 10): {cluster_sizes[:10]}")
        print(f"  Cluster votes (first 10): {cluster_votes[:10]}")
        if cluster_votes:
            print(f"  Max votes: {max(cluster_votes)}, Min vote required: {min_vote}")
            print(f"  Clusters with >= {min_vote} votes: {sum(1 for v in cluster_votes if v >= min_vote)}")
        print(f"  Final ensemble boxes: {len(ensemble_boxes)}")
    
    return ensemble_boxes


def select_best_box_from_cluster(cluster):
    """
    클러스터에서 최고 성능 fold(Fold 3)의 박스를 선택
    좌표 평균 없이 정확한 좌표를 보존
    """
    # Fold 3 (index=3)이 best fold (0.9863)
    # 클러스터 내에 Fold 3 박스가 있으면 우선 선택
    best_fold_idx = 3
    
    for box in cluster:
        if box['fold_idx'] == best_fold_idx:
            return {'points': box['points']}
    
    # Fold 3 박스가 없으면, 가장 높은 weight를 가진 fold 선택
    # Weights: [0.25, 0.15, 0.15, 0.30, 0.15] for Fold [0, 1, 2, 3, 4]
    # Priority order: Fold 3 (0.30) > Fold 0 (0.25) > Others (0.15)
    priority_order = [3, 0, 2, 1, 4]
    
    for fold_idx in priority_order:
        for box in cluster:
            if box['fold_idx'] == fold_idx:
                return {'points': box['points']}
    
    # Fallback: just return first box (should not happen)
    return {'points': cluster[0]['points']}


def interpolate_polygon(points, target_len):
    """
    Polygon을 target_len 개수의 점으로 보간
    """
    if len(points) == target_len:
        return points
    
    # Convert to (x, y) pairs - handle both formats
    if isinstance(points[0], list):
        coords = [(pt[0], pt[1]) for pt in points]
        n_points = target_len
    else:
        coords = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        n_points = target_len // 2
    
    # Calculate cumulative distances
    distances = [0]
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i-1][0]
        dy = coords[i][1] - coords[i-1][1]
        distances.append(distances[-1] + np.sqrt(dx**2 + dy**2))
    
    total_distance = distances[-1]
    
    if total_distance == 0:
        return points
    
    # Interpolate
    new_coords = []
    for i in range(n_points):
        target_dist = (i / n_points) * total_distance
        
        # Find segment
        for j in range(len(distances) - 1):
            if distances[j] <= target_dist <= distances[j+1]:
                # Linear interpolation
                t = (target_dist - distances[j]) / (distances[j+1] - distances[j])
                x = coords[j][0] + t * (coords[j+1][0] - coords[j][0])
                y = coords[j][1] + t * (coords[j+1][1] - coords[j][1])
                new_coords.append((int(round(x)), int(round(y))))
                break
    
    # Return in same format as input
    if isinstance(points[0], list):
        # [[x, y], ...] format
        return [[int(round(x)), int(round(y))] for x, y in new_coords]
    else:
        # [x, y, ...] flat format
        result = []
        for x, y in new_coords:
            result.extend([int(round(x)), int(round(y))])
        return result


def main():
    parser = argparse.ArgumentParser(description='5-Fold K-Fold Ensemble')
    parser.add_argument('--fold0', required=True, help='Fold 0 JSON file')
    parser.add_argument('--fold1', required=True, help='Fold 1 JSON file')
    parser.add_argument('--fold2', required=True, help='Fold 2 JSON file')
    parser.add_argument('--fold3', required=True, help='Fold 3 JSON file')
    parser.add_argument('--fold4', required=True, help='Fold 4 JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--iou-threshold', type=float, default=0.75, 
                       help='IoU threshold for box matching (default: 0.75)')
    parser.add_argument('--min-vote', type=int, default=3,
                       help='Minimum number of folds that must agree (default: 3)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Comma-separated weights for each fold (e.g., "0.25,0.15,0.15,0.30,0.15")')
    
    args = parser.parse_args()
    
    # Parse weights
    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]
        assert len(weights) == 5, "Must provide 5 weights"
    
    # Ensemble
    json_files = [args.fold0, args.fold1, args.fold2, args.fold3, args.fold4]
    
    result = ensemble_kfold_predictions(
        json_files,
        iou_threshold=args.iou_threshold,
        weights=weights,
        min_vote=args.min_vote
    )
    
    # Save result
    print(f"\nSaving to {args.output}")
    output_data = {'images': result}
    with open(args.output, 'w') as f:
        json.dump(output_data, f)
    
    print("Done!")


if __name__ == '__main__':
    main()
