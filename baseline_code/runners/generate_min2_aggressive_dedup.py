#!/usr/bin/env python3
"""min_votes=2 + 낮은 iou_threshold 조합 (올바른 전략)

min_votes를 낮추면 더 많은 박스 포함 → 중복 검출 위험 증가
→ iou_threshold를 **낮춰서** 더 공격적으로 중복 제거해야 함
"""

import json
from pathlib import Path
from collections import OrderedDict, defaultdict
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm
import csv

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


print("=" * 80)
print("min_votes=2 + 공격적 중복 제거 (낮은 iou_threshold)")
print("=" * 80)
print("\n수정된 전략: min_votes↓ → iou_threshold↓ → 중복 제거 강화\n")

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

# min_votes=2 + 낮은 iou_threshold (공격적 중복 제거)
test_configs = [
    {"iou_threshold": 0.45, "min_votes": 2, "tag": "min2_iou45"},
    {"iou_threshold": 0.40, "min_votes": 2, "tag": "min2_iou40"},
    {"iou_threshold": 0.35, "min_votes": 2, "tag": "min2_iou35"},
]

results = []

for config in test_configs:
    print(f"\n{'=' * 60}")
    print(f"테스트: {config['tag']}")
    print(f"  min_votes={config['min_votes']}, iou_threshold={config['iou_threshold']}")
    print(f"{'=' * 60}")
    
    ensemble_preds, stats = ensemble_boxes(
        predictions_by_fold,
        iou_threshold=config["iou_threshold"],
        min_votes=config["min_votes"]
    )
    
    csv_path = Path(f"/data/ephemeral/home/hrnet_ensemble_{config['tag']}.csv")
    convert_to_csv(ensemble_preds, csv_path)
    
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    num_boxes = stats["ensemble_total"]
    
    print(f"\n결과:")
    print(f"  총 박스: {num_boxes:,}개")
    print(f"  파일 크기: {file_size_mb:.1f} MB")
    
    results.append({
        "tag": config["tag"],
        "iou_threshold": config["iou_threshold"],
        "min_votes": config["min_votes"],
        "boxes": num_boxes,
        "file_size_mb": file_size_mb,
    })

# 전체 비교
print(f"\n{'=' * 80}")
print("전체 비교 (min_votes=2)")
print(f"{'=' * 80}\n")
print(f"{'Tag':<20} {'IoU':<8} {'박스 수':<12} {'파일(MB)':<10} {'전략':<20}")
print("-" * 80)

all_results = [
    {"tag": "min2_iou35", "boxes": results[2]["boxes"] if len(results) > 2 else 0, "iou": 0.35, "strategy": "매우 공격적 중복 제거"},
    {"tag": "min2_iou40", "boxes": results[1]["boxes"] if len(results) > 1 else 0, "iou": 0.40, "strategy": "공격적 중복 제거"},
    {"tag": "min2_iou45", "boxes": results[0]["boxes"] if len(results) > 0 else 0, "iou": 0.45, "strategy": "균형 (추천 ⭐)"},
    {"tag": "min2_iou50", "boxes": 46256, "iou": 0.50, "strategy": "기본 (기존)"},
]

for r in all_results:
    if r["boxes"] > 0:
        size = f"{(r['boxes'] * 200 / 1024 / 1024):.1f}"
        print(f"{r['tag']:<20} {r['iou']:<8} {r['boxes']:<12,} {size:<10} {r['strategy']:<20}")

print(f"\n{'=' * 80}")
print("최종 권장")
print(f"{'=' * 80}")
print(f"  1. min2_iou45: 균형잡힌 중복 제거 (추천 ⭐⭐⭐)")
print(f"  2. min2_iou40: 공격적 중복 제거")
print(f"  3. min2_iou35: 매우 공격적 (과하게 제거될 위험)")
print(f"\n결론: iou_threshold를 **낮춰야** 중복 제거 강화!")
print(f"{'=' * 80}")
