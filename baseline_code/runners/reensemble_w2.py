#!/usr/bin/env python3
"""reensemble_w2.py - 기존 Fold 예측 JSON 재활용 + 가중치 v2 앙상블만 재수행.

이미 5-Fold 추론이 완료된 JSON 파일들을 로드하여 앙상블만 다시 수행.
추론 시간 절약 (~5분 → ~30초).
"""

import os
import sys
import csv
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

sys.path.insert(0, "/data/ephemeral/home/baseline_code")

THRESH = 0.225
BOX_THRESH = 0.255
TAG = f"t{THRESH}_b{BOX_THRESH}_w2"

# ============================================================
# 가중치 v2: Fold 4에 더 많은 비중
# ============================================================
fold_weights = {
    4: 0.40,  # Val 0.9837 (최고) ← 0.30에서 상향
    2: 0.25,  # Val 0.9781
    3: 0.15,  # Val 0.9764 ← 0.20에서 하향
    0: 0.12,  # Val 0.9738 ← 0.15에서 하향
    1: 0.08,  # Val 0.9717 (최저) ← 0.10에서 하향
}

print("=" * 80)
print(f"가중치 v2 앙상블 (기존 JSON 재활용)")
print(f"thresh={THRESH}, box_thresh={BOX_THRESH}")
print("=" * 80)
print("\nFold별 가중치 (v2):")
for fold_idx, weight in sorted(fold_weights.items()):
    marker = " ★" if fold_idx == 4 else ""
    print(f"  Fold {fold_idx}: {weight:.2f}{marker}")
print()

# 기존 추론 결과 JSON 로드
base_dir = Path("/data/ephemeral/home/baseline_code/outputs/kfold_ensemble_t0.225_b0.255_w2")
predictions_by_fold = []

for fold_idx in range(5):
    fold_dir = base_dir / f"fold_{fold_idx}"
    json_files = sorted(fold_dir.glob("*.json"))
    if json_files:
        latest_json = json_files[-1]
        with open(latest_json, "r") as f:
            fold_predictions = json.load(f)
        predictions_by_fold.append((fold_idx, fold_predictions))
        n_images = len(fold_predictions.get("images", {}))
        print(f"✓ Fold {fold_idx}: {latest_json.name} ({n_images} images)")
    else:
        print(f"✗ Fold {fold_idx}: JSON 없음")

print(f"\n총 {len(predictions_by_fold)}개 Fold 로드 완료\n")


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


def weighted_average_box(cluster):
    """가중 평균 좌표. 꼭짓점 수가 다르면 best box 폴백."""
    shapes = [b["points"].shape for b in cluster]
    all_same_shape = all(s == shapes[0] for s in shapes)

    if not all_same_shape:
        best_box = max(cluster, key=lambda b: b["weight"])
        return best_box["points"].tolist()

    total_weight = sum(b["weight"] for b in cluster)
    if total_weight == 0:
        return cluster[0]["points"].tolist()

    avg_points = np.zeros_like(cluster[0]["points"], dtype=np.float64)
    for b in cluster:
        avg_points += b["weight"] * b["points"]
    avg_points /= total_weight
    return avg_points.tolist()


def convert_json_to_csv(json_path: Path, output_csv_path: Path) -> int:
    with json_path.open("r") as f:
        data = json.load(f)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if output_csv_path.exists():
        output_csv_path.unlink()
    with output_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "polygons"])
        for filename, content in data["images"].items():
            polygons = []
            for _, word in content["words"].items():
                points = word["points"]
                polygon = " ".join([" ".join(map(str, point)) for point in points])
                polygons.append(polygon)
            writer.writerow([filename, "|".join(polygons)])
    return len(data["images"])


# ============================================================
# 앙상블 수행
# ============================================================
ensemble_predictions = OrderedDict(images=OrderedDict())

all_images = set()
for _, predictions in predictions_by_fold:
    all_images.update(predictions["images"].keys())

print(f"총 {len(all_images)}개 이미지 앙상블 시작\n")

min_votes = 3
iou_threshold = 0.5

stats = defaultdict(int)
fallback_count = 0
avg_count = 0

for image_name in tqdm(sorted(all_images), desc="Ensembling", unit="img"):
    all_boxes = []

    for fold_idx, predictions in predictions_by_fold:
        if image_name not in predictions["images"]:
            continue
        words = predictions["images"][image_name]["words"]
        stats[f"fold_{fold_idx}_boxes"] += len(words)

        for _, word_data in words.items():
            points = word_data["points"]
            all_boxes.append({
                "points": np.array(points, dtype=np.float64),
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
        stats[f"votes_{vote_count}"] += 1

        if vote_count >= min_votes:
            avg_box = weighted_average_box(cluster)
            final_boxes.append(avg_box)
            stats["ensemble_total"] += 1

            # 카운트: 가중평균 vs 폴백
            shapes = [b["points"].shape for b in cluster]
            if all(s == shapes[0] for s in shapes):
                avg_count += 1
            else:
                fallback_count += 1

    words = OrderedDict()
    for idx, points in enumerate(final_boxes):
        word_id = f"{idx + 1:04d}"
        words[word_id] = OrderedDict(points=points)
    ensemble_predictions["images"][image_name] = OrderedDict(words=words)

# 통계
print(f"\n{'=' * 80}")
print("앙상블 통계 (v2)")
print(f"{'=' * 80}")
print("\nFold별 박스:")
for fold_idx in range(5):
    cnt = stats[f"fold_{fold_idx}_boxes"]
    print(f"  Fold {fold_idx}: {cnt:,}개 (weight={fold_weights[fold_idx]:.2f})")

print("\n투표 분포:")
for v in range(1, 6):
    cnt = stats[f"votes_{v}"]
    mark = "✓" if v >= min_votes else "✗"
    print(f"  {v}개 Fold: {cnt:,}개 {mark}")

print(f"\n최종 박스: {stats['ensemble_total']:,}개")
print(f"  가중 평균 적용: {avg_count:,}개")
print(f"  폴백 (best box): {fallback_count:,}개")

# 저장
ensemble_json_path = base_dir / "ensemble_result_w2.json"
with ensemble_json_path.open("w") as f:
    json.dump(ensemble_predictions, f, indent=4)
print(f"\n✓ JSON 저장: {ensemble_json_path}")

csv_path = Path(f"/data/ephemeral/home/hrnet_w44_kfold5_ensemble_improved_P_{TAG}.csv")
num_rows = convert_json_to_csv(ensemble_json_path, csv_path)

file_size_mb = csv_path.stat().st_size / (1024 * 1024)
print(f"\n{'=' * 80}")
print(f"✓ CSV 생성 완료!")
print(f"파일: {csv_path}")
print(f"크기: {file_size_mb:.1f} MB")
print(f"이미지: {num_rows}개")
print(f"{'=' * 80}")
