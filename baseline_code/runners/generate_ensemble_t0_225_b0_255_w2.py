#!/usr/bin/env python3
"""generate_ensemble_t0_225_b0_255_w2.py

HRNet-W44 K-Fold inference + improved ensemble + submission CSV
- thresh=0.225
- box_thresh=0.255
- 가중치 v2: Fold 4에 더 높은 가중치 (0.40)
- 가중 평균 좌표 계산 (best box 선택 → weighted average)

Run:
  cd /data/ephemeral/home/baseline_code
  nohup python -u runners/generate_ensemble_t0_225_b0_255_w2.py > logs/ensemble_w2.log 2>&1 &
"""

import os
import sys
import csv
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import lightning.pytorch as pl
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

sys.path.insert(0, "/data/ephemeral/home/baseline_code")

THRESH = 0.225
BOX_THRESH = 0.255
TAG = f"t{THRESH}_b{BOX_THRESH}_w2"

# Hydra 초기화 정리
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# K-Fold 결과 파일 로드
kfold_results_path = "/data/ephemeral/home/baseline_code/hrnet_w44_1280_optimal_kfold_results_20260207_0458.json"
with open(kfold_results_path, "r") as f:
    kfold_results = json.load(f)

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
print(f"HRNet-W44 K-Fold 앙상블 v2 - thresh={THRESH}, box_thresh={BOX_THRESH}")
print("=" * 80)
print("\n앙상블 설정:")
print(f"  - thresh: {THRESH}")
print(f"  - box_thresh: {BOX_THRESH}")
print("  - 최소 투표 수: 3개 Fold (60% 합의)")
print("  - ★ 가중 평균 좌표 계산 (v2)")
print("  - Fold별 가중치 (v2):")
for fold_idx, weight in sorted(fold_weights.items()):
    val_score = kfold_results["fold_results"][f"fold_{fold_idx}"]["best_score"]
    marker = " ★" if fold_idx == 4 else ""
    print(f"    • Fold {fold_idx}: {weight:.2f} (Val={val_score:.4f}){marker}")
print(f"  - 가중치 합계: {sum(fold_weights.values()):.2f}")
print()

# 각 Fold의 체크포인트 경로
checkpoints = []
for fold_idx in range(5):
    fold_key = f"fold_{fold_idx}"
    checkpoint_path = kfold_results["fold_results"][fold_key]["best_checkpoint"]
    if os.path.exists(checkpoint_path):
        checkpoints.append((fold_idx, checkpoint_path))
        print(f"✓ Fold {fold_idx}: {checkpoint_path}")
    else:
        print(f"✗ Fold {fold_idx}: 체크포인트 없음 - {checkpoint_path}")

print(f"\n총 {len(checkpoints)}개 체크포인트 발견")


def polygon_iou(poly1_points, poly2_points):
    """정교한 폴리곤 IoU 계산"""
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
    """클러스터 내 박스들의 가중 평균 좌표 계산.

    각 fold의 가중치에 따라 좌표를 평균 → 고성능 fold의 좌표에 더 가깝게.
    폴리곤 꼭짓점 수가 다른 경우 → 최고 가중치 fold의 박스 선택 (폴백).
    """
    # 꼭짓점 수가 모두 같은지 확인
    shapes = [b["points"].shape for b in cluster]
    all_same_shape = all(s == shapes[0] for s in shapes)

    if not all_same_shape:
        # 폴백: 최고 가중치 fold의 박스 선택
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


def convert_json_to_csv_no_prompt(json_path: Path, output_csv_path: Path) -> int:
    """Write submission CSV without interactive overwrite prompts."""
    with json_path.open("r") as f:
        data = json.load(f)
    assert "images" in data

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
# 각 Fold별로 예측 수행
# ============================================================
predictions_by_fold = []

for fold_idx, checkpoint_path in checkpoints:
    print(f"\n{'=' * 80}")
    print(f"FOLD {fold_idx} 예측 시작 (thresh={THRESH}, box_thresh={BOX_THRESH})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"가중치: {fold_weights[fold_idx]:.2f}")
    print(f"{'=' * 80}\n")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="predict",
            overrides=[
                "preset=hrnet_w44_1280",
                f"models.head.postprocess.thresh={THRESH}",
                f"models.head.postprocess.box_thresh={BOX_THRESH}",
            ],
        )
        cfg.checkpoint_path = checkpoint_path
        cfg.minified_json = False
        cfg.submission_dir = f"outputs/kfold_ensemble_{TAG}/fold_{fold_idx}"

        from ocr.lightning_modules import get_pl_modules_by_cfg

        model_module, data_module = get_pl_modules_by_cfg(cfg)

        trainer = pl.Trainer(
            logger=False,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=True,
        )

        trainer.predict(model_module, data_module, ckpt_path=checkpoint_path)

        submission_dir = Path(cfg.submission_dir)
        json_files = sorted(submission_dir.glob("*.json"))

        if json_files:
            latest_json = json_files[-1]
            print(f"✓ Fold {fold_idx} 예측 완료: {latest_json}")
            with open(latest_json, "r") as f:
                fold_predictions = json.load(f)
                predictions_by_fold.append((fold_idx, fold_predictions))
        else:
            print(f"✗ Fold {fold_idx} 예측 파일을 찾을 수 없습니다")

print(f"\n{'=' * 80}")
print(f"모든 Fold 예측 완료! 총 {len(predictions_by_fold)}개 결과")
print(f"{'=' * 80}\n")

# ============================================================
# 앙상블 수행 (가중 평균 좌표 v2)
# ============================================================
print("앙상블 수행 중 (가중 평균 좌표 v2)...")

ensemble_predictions = OrderedDict(images=OrderedDict())

all_images = set()
for _, predictions in predictions_by_fold:
    all_images.update(predictions["images"].keys())

print(f"총 {len(all_images)}개 이미지 처리")

stats = {
    "total_boxes_per_fold": defaultdict(int),
    "ensemble_boxes": 0,
    "boxes_with_1_vote": 0,
    "boxes_with_2_votes": 0,
    "boxes_with_3_votes": 0,
    "boxes_with_4_votes": 0,
    "boxes_with_5_votes": 0,
}

for image_name in tqdm(sorted(all_images), desc="Ensembling", unit="img"):
    all_boxes_for_image = []

    for fold_idx, predictions in predictions_by_fold:
        if image_name not in predictions["images"]:
            continue

        words = predictions["images"][image_name]["words"]
        stats["total_boxes_per_fold"][fold_idx] += len(words)

        for _, word_data in words.items():
            points = word_data["points"]
            weight = fold_weights[fold_idx]
            all_boxes_for_image.append(
                {"points": np.array(points, dtype=np.float64),
                 "fold_idx": fold_idx,
                 "weight": weight}
            )

    iou_threshold = 0.5
    min_votes = 3

    final_boxes = []
    used = [False] * len(all_boxes_for_image)

    for i, box1 in enumerate(all_boxes_for_image):
        if used[i]:
            continue

        cluster = [box1]
        cluster_fold_indices = {box1["fold_idx"]}
        used[i] = True

        for j, box2 in enumerate(all_boxes_for_image):
            if used[j]:
                continue
            if box2["fold_idx"] in cluster_fold_indices:
                continue
            if polygon_iou(box1["points"], box2["points"]) > iou_threshold:
                cluster.append(box2)
                cluster_fold_indices.add(box2["fold_idx"])
                used[j] = True

        vote_count = len(cluster)

        if vote_count == 1:
            stats["boxes_with_1_vote"] += 1
        elif vote_count == 2:
            stats["boxes_with_2_votes"] += 1
        elif vote_count == 3:
            stats["boxes_with_3_votes"] += 1
        elif vote_count == 4:
            stats["boxes_with_4_votes"] += 1
        elif vote_count == 5:
            stats["boxes_with_5_votes"] += 1

        if vote_count >= min_votes:
            # ★ v2: 가중 평균 좌표 (기존: best box 선택)
            avg_box = weighted_average_box(cluster)
            final_boxes.append(avg_box)
            stats["ensemble_boxes"] += 1

    words = OrderedDict()
    for idx, points in enumerate(final_boxes):
        word_id = f"{idx + 1:04d}"
        words[word_id] = OrderedDict(points=points)

    ensemble_predictions["images"][image_name] = OrderedDict(words=words)

# 통계 출력
print(f"\n{'=' * 80}")
print("앙상블 통계 (v2 - 가중 평균)")
print(f"{'=' * 80}")
print("\nFold별 원본 박스 수:")
for fold_idx in sorted(stats["total_boxes_per_fold"].keys()):
    count = stats["total_boxes_per_fold"][fold_idx]
    print(f"  Fold {fold_idx}: {count:,}개 (weight={fold_weights[fold_idx]:.2f})")

total_original = sum(stats["total_boxes_per_fold"].values())
print(f"  총합: {total_original:,}개")

print("\n투표 분포:")
included = 0
excluded = 0
for v in range(1, 6):
    cnt = stats[f"boxes_with_{v}_vote{'s' if v > 1 else ''}"]
    mark = "✓" if v >= min_votes else "✗"
    status = "포함" if v >= min_votes else "제외"
    print(f"  {v}개 Fold 합의: {cnt:,}개 ({status} {mark})")
    if v >= min_votes:
        included += cnt
    else:
        excluded += cnt

print(f"\n최종 앙상블 결과:")
print(f"  포함된 박스: {stats['ensemble_boxes']:,}개")
total_all = included + excluded
exclusion_rate = (excluded / total_all * 100) if total_all > 0 else 0
print(f"  제외된 박스: {excluded:,}개 ({exclusion_rate:.1f}%)")

# JSON 저장
ensemble_json_path = Path(f"outputs/kfold_ensemble_{TAG}/ensemble_result.json")
ensemble_json_path.parent.mkdir(parents=True, exist_ok=True)

with ensemble_json_path.open("w") as f:
    json.dump(ensemble_predictions, f, indent=4)

print(f"\n✓ 앙상블 결과 저장: {ensemble_json_path}")

# CSV 변환
print("\nCSV 변환 중...")
csv_output_path = Path(f"/data/ephemeral/home/hrnet_w44_kfold5_ensemble_improved_P_{TAG}.csv")
num_rows = convert_json_to_csv_no_prompt(ensemble_json_path, csv_output_path)

if num_rows:
    file_size_mb = csv_output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 80}")
    print("✓ 앙상블 제출 파일 생성 완료! (v2)")
    print(f"{'=' * 80}")
    print(f"파일: {csv_output_path}")
    print(f"크기: {file_size_mb:.1f} MB")
    print(f"이미지 수: {num_rows}")
    print(f"\n변경 사항 (v1 → v2):")
    print(f"  - Fold 4 가중치: 0.30 → 0.40 (+10%)")
    print(f"  - Fold 3 가중치: 0.20 → 0.15 (-5%)")
    print(f"  - Fold 0 가중치: 0.15 → 0.12 (-3%)")
    print(f"  - Fold 1 가중치: 0.10 → 0.08 (-2%)")
    print(f"  - 좌표 계산: best box 선택 → 가중 평균")
    print(f"{'=' * 80}")
else:
    print("\n✗ CSV 변환 실패")

print("\n✓ 완료!")
