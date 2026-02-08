#!/usr/bin/env python3
"""generate_ensemble_suggested_params.py

제안된 후처리 파라미터로 새 앙상블 생성:
- thresh=0.1505 (기존 0.225 대비 낮음 → 더 많은 박스 검출)
- box_thresh=0.4008 (기존 0.255 대비 높음 → 더 높은 품질 박스만)

주의: 이 조합은 기존 JSON에 없으므로 5-Fold 재추론 필요 (~8분)
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

# 제안된 파라미터
THRESH = 0.1505
BOX_THRESH = 0.4008
TAG = f"t{THRESH}_b{BOX_THRESH}"

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

kfold_results_path = "/data/ephemeral/home/baseline_code/hrnet_w44_1280_optimal_kfold_results_20260207_0458.json"
with open(kfold_results_path, "r") as f:
    kfold_results = json.load(f)

fold_weights = {4: 0.30, 2: 0.25, 3: 0.20, 0: 0.15, 1: 0.10}

print("=" * 80)
print(f"제안된 후처리 파라미터 테스트")
print(f"thresh={THRESH}, box_thresh={BOX_THRESH}")
print("=" * 80)
print("\n⚠️  주의: 기존 JSON과 파라미터가 다르므로 5-Fold 재추론 필요")
print("예상 시간: ~8분\n")

checkpoints = []
for fold_idx in range(5):
    fold_key = f"fold_{fold_idx}"
    checkpoint_path = kfold_results["fold_results"][fold_key]["best_checkpoint"]
    if os.path.exists(checkpoint_path):
        checkpoints.append((fold_idx, checkpoint_path))
        print(f"✓ Fold {fold_idx}: {checkpoint_path}")

print(f"\n총 {len(checkpoints)}개 체크포인트\n")


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


# 5-Fold 예측
predictions_by_fold = []

for fold_idx, checkpoint_path in checkpoints:
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx} 예측 (thresh={THRESH}, box_thresh={BOX_THRESH})")
    print(f"{'=' * 60}\n")

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
            print(f"✓ Fold {fold_idx} 완료")
            with open(latest_json, "r") as f:
                fold_predictions = json.load(f)
                predictions_by_fold.append((fold_idx, fold_predictions))

print(f"\n{'=' * 80}")
print(f"모든 Fold 예측 완료!")
print(f"{'=' * 80}\n")

# 앙상블
ensemble_predictions = OrderedDict(images=OrderedDict())
all_images = set()
for _, predictions in predictions_by_fold:
    all_images.update(predictions["images"].keys())

print(f"앙상블 수행 중... (총 {len(all_images)}개 이미지)\n")

min_votes = 3
iou_threshold = 0.5
stats = defaultdict(int)

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
            best_box = max(cluster, key=lambda b: b["weight"])
            final_boxes.append(best_box["points"].tolist())
            stats["ensemble_total"] += 1

    words = OrderedDict()
    for idx, points in enumerate(final_boxes):
        word_id = f"{idx + 1:04d}"
        words[word_id] = OrderedDict(points=points)
    ensemble_predictions["images"][image_name] = OrderedDict(words=words)

# 통계
print(f"\n{'=' * 80}")
print("앙상블 통계")
print(f"{'=' * 80}")
print("\nFold별 박스:")
for fold_idx in range(5):
    cnt = stats[f"fold_{fold_idx}_boxes"]
    print(f"  Fold {fold_idx}: {cnt:,}개")

print("\n투표 분포:")
for v in range(1, 6):
    cnt = stats[f"votes_{v}"]
    mark = "✓" if v >= min_votes else "✗"
    print(f"  {v}개 Fold: {cnt:,}개 {mark}")

print(f"\n최종 박스: {stats['ensemble_total']:,}개")

# 저장
ensemble_json_path = Path(f"outputs/kfold_ensemble_{TAG}/ensemble_result.json")
ensemble_json_path.parent.mkdir(parents=True, exist_ok=True)
with ensemble_json_path.open("w") as f:
    json.dump(ensemble_predictions, f, indent=4)
print(f"\n✓ JSON 저장: {ensemble_json_path}")

csv_path = Path(f"/data/ephemeral/home/hrnet_ensemble_suggested_{TAG}.csv")
num_rows = convert_json_to_csv(ensemble_json_path, csv_path)

file_size_mb = csv_path.stat().st_size / (1024 * 1024)
print(f"\n{'=' * 80}")
print(f"✓ CSV 생성 완료!")
print(f"파일: {csv_path}")
print(f"크기: {file_size_mb:.1f} MB")
print(f"이미지: {num_rows}개")
print(f"\n파라미터:")
print(f"  thresh: {THRESH}")
print(f"  box_thresh: {BOX_THRESH}")
print(f"  min_votes: {min_votes}")
print(f"  iou_threshold: {iou_threshold}")
print(f"{'=' * 80}")
