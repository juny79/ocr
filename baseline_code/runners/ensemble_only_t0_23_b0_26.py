#!/usr/bin/env python3
"""
이미 생성된 fold prediction JSON들을 ensemble만 수행
"""

import json
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
from shapely.geometry import Polygon
from shapely.validation import make_valid

print("="*80)
print("K-Fold Ensemble Only - thresh=0.23, box_thresh=0.26")
print("="*80)
print()

# 이미 생성된 JSON 파일들
fold_files = [
    'outputs/kfold_ensemble_t0.23_b0.26/fold_0/20260208_010021.json',
    'outputs/kfold_ensemble_t0.23_b0.26/fold_1/20260208_010122.json',
    'outputs/kfold_ensemble_t0.23_b0.26/fold_2/20260208_010222.json',
    'outputs/kfold_ensemble_t0.23_b0.26/fold_3/20260208_010324.json',
    'outputs/kfold_ensemble_t0.23_b0.26/fold_4/20260208_010424.json',
]

# Fold별 가중치
fold_weights = {
    0: 0.15,
    1: 0.10,
    2: 0.25,
    3: 0.20,
    4: 0.30,
}

predictions_by_fold = []

print("Loading fold predictions...")
for fold_idx, fpath in enumerate(fold_files):
    with open(fpath, 'r') as f:
        predictions = json.load(f)
    predictions_by_fold.append((fold_idx, predictions))
    img_count = len(predictions['images'])
    print(f"  Fold {fold_idx}: {img_count} images")

print()
print("Ensemble settings:")
print(f"  Min votes: 3/5 folds")
print(f"  IoU threshold: 0.5")
print()

# 앙상블 수행
ensemble_predictions = OrderedDict(images=OrderedDict())

all_images = set()
for fold_idx, predictions in predictions_by_fold:
    all_images.update(predictions['images'].keys())

print(f"Processing {len(all_images)} images...")

stats = {
    'total_boxes_per_fold': defaultdict(int),
    'ensemble_boxes': 0,
    'boxes_with_1_vote': 0,
    'boxes_with_2_votes': 0,
    'boxes_with_3_votes': 0,
    'boxes_with_4_votes': 0,
    'boxes_with_5_votes': 0,
}

def polygon_iou(poly1_points, poly2_points):
    """Polygon IoU calculation"""
    try:
        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)
        
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

# Ensemble each image
for image_name in sorted(all_images):
    all_boxes_for_image = []
    
    for fold_idx, predictions in predictions_by_fold:
        if image_name in predictions['images']:
            words = predictions['images'][image_name]['words']
            stats['total_boxes_per_fold'][fold_idx] += len(words)
            
            for word_id, word_data in words.items():
                points = word_data['points']
                weight = fold_weights[fold_idx]
                all_boxes_for_image.append({
                    'points': np.array(points),
                    'fold_idx': fold_idx,
                    'weight': weight
                })
    
    iou_threshold = 0.5
    min_votes = 3
    
    final_boxes = []
    used = [False] * len(all_boxes_for_image)
    
    for i, box1 in enumerate(all_boxes_for_image):
        if used[i]:
            continue
        
        similar_boxes = [box1]
        similar_indices = [i]
        
        for j, box2 in enumerate(all_boxes_for_image):
            if i != j and not used[j]:
                iou = polygon_iou(box1['points'], box2['points'])
                if iou > iou_threshold:
                    similar_boxes.append(box2)
                    similar_indices.append(j)
        
        vote_count = len(similar_boxes)
        
        if vote_count == 1:
            stats['boxes_with_1_vote'] += 1
        elif vote_count == 2:
            stats['boxes_with_2_votes'] += 1
        elif vote_count == 3:
            stats['boxes_with_3_votes'] += 1
        elif vote_count == 4:
            stats['boxes_with_4_votes'] += 1
        elif vote_count == 5:
            stats['boxes_with_5_votes'] += 1
        
        if vote_count >= min_votes:
            # Use highest weighted box's polygon
            best_box = max(similar_boxes, key=lambda b: b['weight'])
            final_boxes.append(best_box['points'].tolist())
            stats['ensemble_boxes'] += 1
            
            for idx in similar_indices:
                used[idx] = True
    
    words = OrderedDict()
    for idx, points in enumerate(final_boxes):
        word_id = f"{idx+1:04d}"
        words[word_id] = OrderedDict(points=points)
    
    ensemble_predictions['images'][image_name] = OrderedDict(words=words)

# Statistics
print(f"\n{'='*80}")
print("Ensemble Statistics")
print(f"{'='*80}")
print(f"\nOriginal boxes per fold:")
for fold_idx in sorted(stats['total_boxes_per_fold'].keys()):
    count = stats['total_boxes_per_fold'][fold_idx]
    print(f"  Fold {fold_idx}: {count:,}")

total_original = sum(stats['total_boxes_per_fold'].values())
print(f"  Total: {total_original:,}")

print(f"\nVoting distribution:")
print(f"  1 vote: {stats['boxes_with_1_vote']:,} (excluded ✗)")
print(f"  2 votes: {stats['boxes_with_2_votes']:,} (excluded ✗)")
print(f"  3 votes: {stats['boxes_with_3_votes']:,} (included ✓)")
print(f"  4 votes: {stats['boxes_with_4_votes']:,} (included ✓)")
print(f"  5 votes: {stats['boxes_with_5_votes']:,} (included ✓)")

print(f"\nFinal ensemble:")
print(f"  Included: {stats['ensemble_boxes']:,}")
excluded = total_original - stats['ensemble_boxes']
exclusion_rate = (excluded / total_original * 100) if total_original > 0 else 0
print(f"  Excluded: {excluded:,} ({exclusion_rate:.1f}%)")

# Save ensemble result
ensemble_json_path = Path('outputs/kfold_ensemble_t0.23_b0.26/ensemble_result.json')
ensemble_json_path.parent.mkdir(parents=True, exist_ok=True)

with ensemble_json_path.open('w') as f:
    json.dump(ensemble_predictions, f, indent=4)

print(f"\n✓ Ensemble saved: {ensemble_json_path}")

# Convert to CSV
print("\nConverting to CSV...")
from ocr.utils.convert_submission import convert_json_to_csv

csv_output_path = '/data/ephemeral/home/hrnet_w44_kfold5_ensemble_t0.23_b0.26_NEW.csv'
result = convert_json_to_csv(str(ensemble_json_path), csv_output_path)

if result:
    num_rows, output_file = result
    print(f"\n{'='*80}")
    print(f"✓ CSV 생성 완료!")
    print(f"{'='*80}")
    print(f"File: {output_file}")
    print(f"Images: {num_rows}")
    print(f"\nParameters:")
    print(f"  thresh: 0.23")
    print(f"  box_thresh: 0.26")
    print(f"\nExpected performance:")
    print(f"  Hmean: 0.9837-0.9842 (similar or slightly better than 0.9840)")
    print(f"{'='*80}")
else:
    print("\n✗ CSV conversion failed")

print("\n✓ Done!")
