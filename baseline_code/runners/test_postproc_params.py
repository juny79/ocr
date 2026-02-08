#!/usr/bin/env python3
"""test_postproc_params.py - 다양한 후처리 파라미터 조합 빠른 테스트

기존 5-Fold 예측 JSON을 재활용하여 다양한 후처리 파라미터 조합을 테스트.
- thresh, box_thresh 조합
- unclip_ratio는 코드 변경 필요하므로 제외
- 앙상블 파라미터 (min_votes, iou_threshold) 조합

목표: 리더보드 제출 전 빠른 파라미터 최적화 (재학습 없이)
"""

import json
from pathlib import Path
from collections import OrderedDict, defaultdict
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm
import csv

# Fold별 가중치 (기존 최적값)
fold_weights = {4: 0.30, 2: 0.25, 3: 0.20, 0: 0.15, 1: 0.10}


def polygon_iou(poly1_points, poly2_points):
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
    except Exception:
        return 0.0


def ensemble_boxes(predictions_by_fold, iou_threshold=0.5, min_votes=3):
    """앙상블 수행 - 이미 thresh/box_thresh 필터링된 예측값 사용"""
    ensemble_predictions = OrderedDict(images=OrderedDict())

    all_images = set()
    for _, predictions in predictions_by_fold:
        all_images.update(predictions["images"].keys())

    stats = defaultdict(int)

    for image_name in tqdm(sorted(all_images), desc="Ensembling", unit="img", leave=False):
        all_boxes = []

        for fold_idx, predictions in predictions_by_fold:
            if image_name not in predictions["images"]:
                continue
            words = predictions["images"][image_name]["words"]
            for _, word_data in words.items():
                points = word_data["points"]
                all_boxes.append({
                    "points": points,
                    "fold_idx": fold_idx,
                    "weight": fold_weights[fold_idx],
                })

        final_boxes = []
        used = [False] * len(all_boxes)

        for i, box1 in enumerate(all_boxes):
            if used[i]:
                continue
            cluster = [box1]
            cluster_folds = {box1["fold_idx"]}
            used[i] = True

            for j, box2 in enumerate(all_boxes):
                if used[j] or box2["fold_idx"] in cluster_folds:
                    continue
                if polygon_iou(box1["points"], box2["points"]) > iou_threshold:
                    cluster.append(box2)
                    cluster_folds.add(box2["fold_idx"])
                    used[j] = True

            vote_count = len(cluster)
            if vote_count >= min_votes:
                best_box = max(cluster, key=lambda b: b["weight"])
                final_boxes.append(best_box["points"])
                stats["ensemble_total"] += 1

        words = OrderedDict()
        for idx, points in enumerate(final_boxes):
            word_id = f"{idx + 1:04d}"
            words[word_id] = OrderedDict(points=points)
        ensemble_predictions["images"][image_name] = OrderedDict(words=words)

    return ensemble_predictions, stats


def convert_to_csv(ensemble_predictions, output_path):
    """JSON → CSV 변환"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "polygons"])
        for filename, content in ensemble_predictions["images"].items():
            polygons = []
            for _, word in content["words"].items():
                points = word["points"]
                polygon = " ".join([" ".join(map(str, point)) for point in points])
                polygons.append(polygon)
            writer.writerow([filename, "|".join(polygons)])


# ============================================================
# 테스트 시나리오
# ============================================================

print("=" * 80)
print("후처리 파라미터 최적화 테스트")
print("=" * 80)
print("\n현재 기존 JSON 재활용 (t0.225/b0.255 기반)")
print("테스트 대상: 앙상블 파라미터 (iou_threshold, min_votes)\n")

# 기존 예측 로드
base_dir = Path("/data/ephemeral/home/baseline_code/outputs/kfold_ensemble_t0.225_b0.255_w2")
predictions_by_fold = []

for fold_idx in range(5):
    fold_dir = base_dir / f"fold_{fold_idx}"
    json_files = sorted(fold_dir.glob("*.json"))
    if json_files:
        with open(json_files[-1], "r") as f:
            fold_predictions = json.load(f)
        predictions_by_fold.append((fold_idx, fold_predictions))
        print(f"✓ Fold {fold_idx} 로드")

print(f"\n테스트 시작...\n")

# 테스트 조합
test_configs = [
    # 현재 설정
    {"iou_threshold": 0.5, "min_votes": 3, "tag": "baseline"},
    
    # min_votes 변경
    {"iou_threshold": 0.5, "min_votes": 2, "tag": "min2_iou50"},  # 더 많은 박스 (Recall↑)
    {"iou_threshold": 0.5, "min_votes": 4, "tag": "min4_iou50"},  # 더 적은 박스 (Precision↑)
    
    # iou_threshold 변경
    {"iou_threshold": 0.4, "min_votes": 3, "tag": "min3_iou40"},  # 더 느슨한 매칭
    {"iou_threshold": 0.6, "min_votes": 3, "tag": "min3_iou60"},  # 더 엄격한 매칭
    
    # 조합 최적화
    {"iou_threshold": 0.45, "min_votes": 3, "tag": "min3_iou45"},
    {"iou_threshold": 0.55, "min_votes": 3, "tag": "min3_iou55"},
]

results = []

for config in test_configs:
    print(f"\n{'=' * 60}")
    print(f"테스트: {config['tag']}")
    print(f"  iou_threshold={config['iou_threshold']}, min_votes={config['min_votes']}")
    print(f"{'=' * 60}")
    
    ensemble_preds, stats = ensemble_boxes(
        predictions_by_fold,
        iou_threshold=config["iou_threshold"],
        min_votes=config["min_votes"]
    )
    
    # CSV 생성
    csv_path = Path(f"/data/ephemeral/home/hrnet_ensemble_{config['tag']}.csv")
    convert_to_csv(ensemble_preds, csv_path)
    
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    num_boxes = stats["ensemble_total"]
    
    print(f"\n결과:")
    print(f"  총 박스: {num_boxes:,}개")
    print(f"  파일 크기: {file_size_mb:.1f} MB")
    print(f"  CSV: {csv_path.name}")
    
    results.append({
        "tag": config["tag"],
        "iou_threshold": config["iou_threshold"],
        "min_votes": config["min_votes"],
        "boxes": num_boxes,
        "file_size_mb": file_size_mb,
        "csv_path": str(csv_path),
    })

# 요약
print(f"\n{'=' * 80}")
print("테스트 결과 요약")
print(f"{'=' * 80}\n")
print(f"{'Tag':<20} {'IoU':<6} {'MinV':<5} {'Boxes':<10} {'Size(MB)':<10}")
print("-" * 80)
for r in results:
    print(f"{r['tag']:<20} {r['iou_threshold']:<6} {r['min_votes']:<5} {r['boxes']:<10,} {r['file_size_mb']:<10.1f}")

print(f"\n권장 제출 순서:")
print(f"  1. baseline (현재 설정)")
print(f"  2. min2_iou50 (Recall 향상 기대)")
print(f"  3. min4_iou50 (Precision 향상 기대)")
print(f"  4. 나머지는 1-3의 결과 보고 결정")
print(f"{'=' * 80}")
