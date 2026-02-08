#!/usr/bin/env python3
"""min_votes=4 보수적 전략

리더보드 결과:
- min_votes=2: Precision -1.4%p (실패)
- min_votes=4: Precision 극대화 전략
- 높은 신뢰도만 채택 → FP 최소화
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


def ensemble_boxes(predictions_by_fold, iou_threshold=0.5, min_votes=4):
    ensemble_predictions = OrderedDict(images=OrderedDict())
    all_images = set()
    for _, predictions in predictions_by_fold:
        all_images.update(predictions["images"].keys())

    for image_name in tqdm(sorted(all_images), desc="Ensembling", unit="img", leave=False):
        all_boxes = []

        for fold_idx, predictions in predictions_by_fold:
            if image_name not in predictions["images"]:
                continue
            words = predictions["images"][image_name]["words"]
            for _, word_data in words.items():
                points = word_data["points"]
                all_boxes.append({"points": points, "fold": fold_idx})

        if not all_boxes:
            ensemble_predictions["images"][image_name] = {"words": {}}
            continue

        used = [False] * len(all_boxes)
        clusters = []

        for i, box in enumerate(all_boxes):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True

            for j in range(i + 1, len(all_boxes)):
                if used[j]:
                    continue
                iou = polygon_iou(box["points"], all_boxes[j]["points"])
                if iou >= iou_threshold:
                    cluster.append(j)
                    used[j] = True

            clusters.append(cluster)

        final_words = {}
        word_id = 0

        for cluster in clusters:
            fold_votes = set(all_boxes[idx]["fold"] for idx in cluster)
            if len(fold_votes) < min_votes:
                continue

            total_weight = 0
            avg_points = np.zeros((4, 2))

            for idx in cluster:
                box = all_boxes[idx]
                weight = fold_weights[box["fold"]]
                points = np.array(box["points"])
                
                if points.shape == (4, 2):
                    avg_points += points * weight
                    total_weight += weight

            if total_weight > 0:
                avg_points /= total_weight
                final_words[str(word_id)] = {
                    "points": avg_points.tolist(),
                    "transcription": "###",
                    "orientation": "Horizontal"
                }
                word_id += 1

        ensemble_predictions["images"][image_name] = {"words": final_words}

    return ensemble_predictions


def save_submission(predictions, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "points"])

        for image_name, image_data in predictions["images"].items():
            words = image_data["words"]
            if not words:
                points_str = "[]"
            else:
                all_points = []
                for word_data in words.values():
                    points = word_data["points"]
                    flat_points = [coord for point in points for coord in point]
                    all_points.append(flat_points)
                points_str = str(all_points)
            writer.writerow([image_name, points_str])


def main():
    base_dir = Path("/data/ephemeral/home/baseline_code")
    json_dir = base_dir / "outputs/kfold_ensemble_t0.225_b0.255_w2"
    output_dir = Path("/data/ephemeral/home")

    predictions_by_fold = []
    for fold_idx in range(5):
        fold_dir = json_dir / f"fold_{fold_idx}"
        json_files = list(fold_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files in {fold_dir}")
        json_path = json_files[0]
        print(f"Loading {json_path}...")
        with open(json_path) as f:
            predictions = json.load(f)
        predictions_by_fold.append((fold_idx, predictions))

    print("\n=== min_votes=4 보수적 전략 (Precision 극대화) ===\n")

    configs = [
        {"min_votes": 4, "iou_threshold": 0.50, "tag": "min4_iou50"},
        {"min_votes": 4, "iou_threshold": 0.48, "tag": "min4_iou48"},
        {"min_votes": 4, "iou_threshold": 0.45, "tag": "min4_iou45"},
    ]

    for config in configs:
        min_votes = config["min_votes"]
        iou_threshold = config["iou_threshold"]
        tag = config["tag"]

        print(f"\n{'='*50}")
        print(f"Processing: {tag}")
        print(f"min_votes={min_votes}, iou_threshold={iou_threshold}")
        print(f"{'='*50}")

        ensemble_pred = ensemble_boxes(
            predictions_by_fold,
            iou_threshold=iou_threshold,
            min_votes=min_votes
        )

        total_boxes = sum(
            len(img_data["words"])
            for img_data in ensemble_pred["images"].values()
        )

        output_csv = output_dir / f"hrnet_ensemble_{tag}.csv"
        save_submission(ensemble_pred, output_csv)

        file_size_mb = output_csv.stat().st_size / (1024 * 1024)
        print(f"\n✓ {tag}: {total_boxes:,}개 ({file_size_mb:.1f} MB)")

    print("\n" + "="*50)
    print("완료! min_votes=4 조합:")
    print("="*50)
    for config in configs:
        output_csv = output_dir / f"hrnet_ensemble_{config['tag']}.csv"
        if output_csv.exists():
            file_size_mb = output_csv.stat().st_size / (1024 * 1024)
            print(f"  {output_csv.name:30s} ({file_size_mb:.1f} MB)")

    print("\n특징:")
    print("  - 4개 이상 Fold 동의 필요 (매우 높은 신뢰도)")
    print("  - False Positive 최소화")
    print("  - Precision 극대화 (0.990+ 예상)")
    print("  - Recall 소폭 하락 가능")


if __name__ == "__main__":
    main()
