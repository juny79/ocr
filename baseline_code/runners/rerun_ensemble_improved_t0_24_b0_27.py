#!/usr/bin/env python3
"""Re-run Feb7 improved ensemble pipeline for t0.24/b0.27.

This script intentionally avoids re-running inference (multi-hour).
It reuses the fold prediction JSONs produced during the original Feb7 run
under `outputs/kfold_ensemble_improved_temp/` and regenerates:
- an ensembled JSON result
- a submission CSV

Output:
- JSON: outputs/kfold_ensemble_improved_rerun_t0.24_b0.27/ensemble_result.json
- CSV : /data/ephemeral/home/hrnet_w44_kfold5_ensemble_improved_P_t0.24_b0.27.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.validation import make_valid


FOLD_WEIGHTS = {
    4: 0.30,
    2: 0.25,
    3: 0.20,
    0: 0.15,
    1: 0.10,
}


def polygon_iou(poly1_points: np.ndarray, poly2_points: np.ndarray) -> float:
    """Polygon IoU (Shapely). Falls back to AABB IoU on errors."""
    try:
        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)

        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)

        if poly1.is_empty or poly2.is_empty:
            return 0.0

        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return float(inter_area / union_area) if union_area > 0 else 0.0
    except Exception:
        try:
            poly1_arr = np.array(poly1_points)
            poly2_arr = np.array(poly2_points)

            x1_min, y1_min = poly1_arr.min(axis=0)
            x1_max, y1_max = poly1_arr.max(axis=0)
            x2_min, y2_min = poly2_arr.min(axis=0)
            x2_max, y2_max = poly2_arr.max(axis=0)

            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)

            if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
                return 0.0

            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            return float(inter_area / union_area) if union_area > 0 else 0.0
        except Exception:
            return 0.0


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def pick_latest_json(fold_dir: Path) -> Path:
    json_files = list(fold_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No json files in {fold_dir}")
    return max(json_files, key=lambda p: p.stat().st_mtime)


def overwrite_convert_json_to_csv(json_path: Path, output_csv_path: Path) -> None:
    import pandas as pd

    data = load_json(json_path)
    assert "images" in data

    rows = []
    for filename, content in data["images"].items():
        polygons = []
        for _, word in content["words"].items():
            points = word["points"]
            polygon = " ".join([" ".join(map(str, point)) for point in points])
            polygons.append(polygon)

        rows.append([filename, "|".join(polygons)])

    df = pd.DataFrame(rows, columns=["filename", "polygons"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)


def main() -> None:
    base_dir = Path("outputs/kfold_ensemble_improved_temp")
    out_dir = Path("outputs/kfold_ensemble_improved_rerun_t0.24_b0.27")
    out_json = out_dir / "ensemble_result.json"
    out_csv = Path("/data/ephemeral/home/hrnet_w44_kfold5_ensemble_improved_P_t0.24_b0.27.csv")

    print("=" * 80)
    print("Re-run improved ensemble (t0.24/b0.27) from existing fold JSONs")
    print("=" * 80)

    predictions_by_fold: list[tuple[int, dict]] = []
    picked_paths: dict[int, Path] = {}

    for fold_idx in range(5):
        fold_dir = base_dir / f"fold_{fold_idx}"
        latest = pick_latest_json(fold_dir)
        picked_paths[fold_idx] = latest
        pred = load_json(latest)
        print(f"✓ Fold {fold_idx}: {latest} (images={len(pred.get('images', {}))})")
        predictions_by_fold.append((fold_idx, pred))

    # Collect all images
    all_images = set()
    for _, pred in predictions_by_fold:
        all_images.update(pred["images"].keys())

    print(f"Total images: {len(all_images)}")

    stats = {
        "total_boxes_per_fold": defaultdict(int),
        "ensemble_boxes": 0,
        "boxes_with_1_vote": 0,
        "boxes_with_2_vote": 0,
        "boxes_with_3_vote": 0,
        "boxes_with_4_vote": 0,
        "boxes_with_5_vote": 0,
    }

    ensemble_predictions = OrderedDict(images=OrderedDict())

    iou_threshold = 0.5
    min_votes = 3

    for image_name in tqdm(sorted(all_images), desc="Ensembling", unit="img"):
        all_boxes_for_image = []

        for fold_idx, pred in predictions_by_fold:
            words = pred["images"][image_name]["words"]
            stats["total_boxes_per_fold"][fold_idx] += len(words)

            for _, word_data in words.items():
                points = word_data["points"]
                all_boxes_for_image.append(
                    {
                        "points": np.array(points),
                        "fold_idx": fold_idx,
                        "weight": FOLD_WEIGHTS[fold_idx],
                    }
                )

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
                stats["boxes_with_2_vote"] += 1
            elif vote_count == 3:
                stats["boxes_with_3_vote"] += 1
            elif vote_count == 4:
                stats["boxes_with_4_vote"] += 1
            elif vote_count == 5:
                stats["boxes_with_5_vote"] += 1

            if vote_count >= min_votes:
                best_box = max(cluster, key=lambda b: b["weight"])
                final_boxes.append(best_box["points"])
                stats["ensemble_boxes"] += 1

        words_out = OrderedDict()
        for idx, box in enumerate(final_boxes):
            box_int = [[int(round(x)), int(round(y))] for x, y in box]
            words_out[f"{idx + 1:04}"] = OrderedDict(points=box_int)

        ensemble_predictions["images"][image_name] = OrderedDict(words=words_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(ensemble_predictions, f, indent=4)

    print(f"\n✓ Saved JSON: {out_json}")

    total_original = sum(stats["total_boxes_per_fold"].values())
    excluded = total_original - stats["ensemble_boxes"]
    exclusion_rate = (excluded / total_original * 100) if total_original > 0 else 0.0
    print("\n" + "=" * 80)
    print("Ensemble statistics")
    print("=" * 80)
    for fold_idx in range(5):
        print(f"Fold {fold_idx}: {stats['total_boxes_per_fold'][fold_idx]:,} boxes")
    print(f"Total original: {total_original:,}")
    print(
        "Votes: "
        f"1={stats['boxes_with_1_vote']:,}, "
        f"2={stats['boxes_with_2_vote']:,}, "
        f"3={stats['boxes_with_3_vote']:,}, "
        f"4={stats['boxes_with_4_vote']:,}, "
        f"5={stats['boxes_with_5_vote']:,}"
    )
    print(f"Final included: {stats['ensemble_boxes']:,}")
    print(f"Final excluded: {excluded:,} ({exclusion_rate:.1f}%)")

    # Overwrite output CSV (non-interactive)
    if out_csv.exists():
        out_csv.unlink()
    overwrite_convert_json_to_csv(out_json, out_csv)

    size_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"✓ Saved CSV : {out_csv} ({size_mb:.2f} MB)")

    # Quick sanity: point count of first word
    first_img = next(iter(ensemble_predictions["images"].keys()))
    first_word = next(iter(ensemble_predictions["images"][first_img]["words"].values()))
    print(f"Sanity: first polygon points={len(first_word['points'])}")


if __name__ == "__main__":
    main()
