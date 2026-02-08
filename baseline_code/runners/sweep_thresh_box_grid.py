#!/usr/bin/env python3
"""sweep_thresh_box_grid.py

실제 점수(CLEval) 기반으로 thresh/box_thresh 미세조정을 그리드 서치.

동작:
- 각 (thresh, box_thresh) 조합마다
  - 5개 fold checkpoint로 validation set(기본: val.json)을 예측 (predict 단계, export 비활성)
  - improved voting ensemble 수행 (polygon IoU 기반)
  - CLEvalMetric으로 P/R/Hmean 계산
- 결과를 JSONL/CSV로 저장

실행(터미널 실시간 출력 추천):
  cd /data/ephemeral/home/baseline_code
  python -u runners/sweep_thresh_box_grid.py \
    --preset hrnet_w44_1280 \
    --kfold-results-json /data/ephemeral/home/baseline_code/hrnet_w44_1280_optimal_kfold_results_20260207_0458.json \
    --val-json /data/ephemeral/home/data/datasets/jsons/val.json \
    --image-dir /data/ephemeral/home/data/datasets/images/all \
    --thresh 0.225 0.230 0.235 \
    --box-thresh 0.255 0.260 0.265 \
    --min-votes 3 --iou-threshold 0.5 \
    --out-dir outputs/postproc_sweep/tgrid_20260208 \
    2>&1 | tee logs/postproc_sweep_tgrid_20260208.log

주의:
- 조합 수 × 5 folds 만큼 예측을 수행하므로 시간이 오래 걸릴 수 있습니다.
- compute 절약을 위해 3x3 같은 작은 그리드부터 시작 권장.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import lightning.pytorch as pl
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# Make `ocr` importable when executed from repo root
sys.path.append(os.getcwd())

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.metrics import CLEvalMetric  # noqa: E402


@dataclass(frozen=True)
class SweepResult:
    thresh: float
    box_thresh: float
    precision: float
    recall: float
    hmean: float
    elapsed_sec: float
    per_fold_time_sec: Dict[int, float]


def _load_kfold_checkpoints(kfold_results_json: str) -> Dict[int, str]:
    with open(kfold_results_json, "r") as f:
        data = json.load(f)

    # Expected shape:
    # data['fold_results']['fold_0']['best_checkpoint']
    fold_results = data.get("fold_results") or {}
    checkpoints: Dict[int, str] = {}

    for fold_key, fold_info in fold_results.items():
        # fold_key like 'fold_0'
        if not fold_key.startswith("fold_"):
            continue
        try:
            fold_idx = int(fold_key.split("_")[1])
        except Exception:
            continue
        ckpt = fold_info.get("best_checkpoint") or fold_info.get("checkpoint")
        if not ckpt:
            continue
        checkpoints[fold_idx] = ckpt

    if not checkpoints:
        raise ValueError(
            "Failed to parse checkpoints from kfold results JSON. "
            "Expected fold_results.fold_i.best_checkpoint"
        )

    return checkpoints


def _polygon_iou(poly1_points: np.ndarray, poly2_points: np.ndarray) -> float:
    try:
        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)

        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        return 0.0


def _ensemble_improved(
    predictions_by_fold: Dict[int, Dict[str, List[List[List[float]]]]],
    fold_weights: Dict[int, float],
    min_votes: int,
    iou_threshold: float,
) -> Dict[str, List[np.ndarray]]:
    """Return ensembled polygons per image.

    predictions_by_fold[fold_idx][img_name] = list of polygons
    polygon = list of [x, y] points
    """

    all_image_names: List[str] = []
    for fold_idx in sorted(predictions_by_fold.keys()):
        all_image_names.extend(list(predictions_by_fold[fold_idx].keys()))
    # Keep deterministic order, unique
    image_names = list(OrderedDict.fromkeys(all_image_names).keys())

    ensembled: Dict[str, List[np.ndarray]] = {}

    for img_name in tqdm(image_names, desc="Ensemble", leave=False):
        all_boxes: List[Dict[str, Any]] = []

        for fold_idx, pred_map in predictions_by_fold.items():
            if img_name not in pred_map:
                continue
            weight = fold_weights.get(fold_idx, 0.0)
            for poly in pred_map[img_name]:
                all_boxes.append(
                    {
                        "points": np.asarray(poly, dtype=np.float32),
                        "fold": fold_idx,
                        "weight": float(weight),
                    }
                )

        clusters: List[List[Dict[str, Any]]] = []
        used = set()

        for i, box1 in enumerate(all_boxes):
            if i in used:
                continue

            cluster = [box1]
            used.add(i)

            for j, box2 in enumerate(all_boxes):
                if j in used or i == j:
                    continue
                if _polygon_iou(box1["points"], box2["points"]) > iou_threshold:
                    cluster.append(box2)
                    used.add(j)

            clusters.append(cluster)

        final_boxes: List[np.ndarray] = []
        for cluster in clusters:
            folds_in_cluster = {b["fold"] for b in cluster}
            if len(folds_in_cluster) < min_votes:
                continue

            # Improved rule: at most one box per fold per cluster
            best_per_fold: Dict[int, Dict[str, Any]] = {}
            for b in cluster:
                f = b["fold"]
                if (f not in best_per_fold) or (b["weight"] > best_per_fold[f]["weight"]):
                    best_per_fold[f] = b

            # Pick the highest weight among fold winners
            best_box = max(best_per_fold.values(), key=lambda b: b["weight"])
            final_boxes.append(best_box["points"])

        ensembled[img_name] = final_boxes

    return ensembled


def _load_gt_quads(val_json_path: str) -> Dict[str, List[np.ndarray]]:
    with open(val_json_path, "r") as f:
        data = json.load(f)

    gt: Dict[str, List[np.ndarray]] = {}
    images = data.get("images") or {}
    for filename, image_info in images.items():
        words = (image_info or {}).get("words") or {}
        polys: List[np.ndarray] = []
        for _, w in words.items():
            pts = (w or {}).get("points")
            if not pts:
                continue
            polys.append(np.asarray(pts, dtype=np.float32).reshape(-1))
        gt[filename] = polys
    return gt


def _eval_cleval(ensembled: Dict[str, List[np.ndarray]], gt_quads: Dict[str, List[np.ndarray]]) -> Tuple[float, float, float]:
    metric = CLEvalMetric()
    recalls: List[float] = []
    precisions: List[float] = []
    hmeans: List[float] = []

    for filename, gt_polys in tqdm(gt_quads.items(), desc="CLEval", leave=False):
        pred_polys = ensembled.get(filename, [])

        det_quads = [poly.reshape(-1).tolist() for poly in pred_polys]
        gt_quads = [p.reshape(-1) for p in gt_polys]

        metric(det_quads, gt_quads)
        out = metric.compute()
        recalls.append(float(out["det_r"].cpu().numpy()))
        precisions.append(float(out["det_p"].cpu().numpy()))
        hmeans.append(float(out["det_h"].cpu().numpy()))
        metric.reset()

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(hmeans))


def _predict_fold_on_val(
    preset: str,
    checkpoint_path: str,
    thresh: float,
    box_thresh: float,
    val_json: str,
    image_dir: str,
    devices: int,
    accelerator: str,
) -> Dict[str, List[List[List[float]]]]:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="predict",
            overrides=[
                f"preset={preset}",
                # run predict over validation subset
                f"datasets.predict_dataset.image_path={image_dir}",
                f"datasets.predict_dataset.annotation_path={val_json}",
                # Postprocess tuning (inference-time)
                f"models.head.postprocess.thresh={thresh}",
                f"models.head.postprocess.box_thresh={box_thresh}",
                # Disable export so we can read in-memory outputs
                "+disable_predict_export=true",
                # Faster/cleaner runs
                "minified_json=true",
            ],
        )

    # checkpoint paths often contain '=' (epoch=..), which is not safe in Hydra overrides.
    cfg.checkpoint_path = checkpoint_path

    pl.seed_everything(int(getattr(cfg, "seed", 42)), workers=True)
    model_module, data_module = get_pl_modules_by_cfg(cfg)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
    )

    trainer.predict(model_module, data_module, ckpt_path=checkpoint_path)

    # Read and detach predictions then clear to free memory
    pred_map = dict(model_module.predict_step_outputs)
    model_module.predict_step_outputs.clear()

    # Convert numpy arrays to python lists for downstream ensembling
    cleaned: Dict[str, List[List[List[float]]]] = {}
    for filename, polys in pred_map.items():
        cleaned[filename] = [np.asarray(p, dtype=np.float32).tolist() for p in polys]

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="hrnet_w44_1280")
    parser.add_argument("--kfold-results-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--thresh", type=float, nargs="+", required=True)
    parser.add_argument("--box-thresh", type=float, nargs="+", required=True)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Run only the specified fold indices (default: all folds found in kfold-results-json)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit sweep to first N images from val-json (for faster iteration)",
    )
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fold_predictions").mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"results_{ts}.jsonl"
    csv_path = out_dir / f"results_{ts}.csv"

    checkpoints = _load_kfold_checkpoints(args.kfold_results_json)
    fold_indices = sorted(checkpoints.keys())
    if args.folds is not None and len(args.folds) > 0:
        fold_indices = [f for f in fold_indices if f in set(args.folds)]
        if not fold_indices:
            raise ValueError("--folds did not match any fold indices from kfold-results-json")

    # Deterministic weights (same as improved ensemble script)
    fold_weights = {
        4: 0.30,
        2: 0.25,
        3: 0.20,
        0: 0.15,
        1: 0.10,
    }

    # Optionally shrink validation set for quick sweeps
    val_json_for_run = args.val_json
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be a positive integer")

        with open(args.val_json, "r") as f:
            raw = json.load(f)
        images = raw.get("images") or {}
        selected_keys = list(images.keys())[: args.max_images]
        raw["images"] = {k: images[k] for k in selected_keys}

        val_json_for_run = str(out_dir / f"val_subset_{args.max_images}.json")
        with open(val_json_for_run, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)

    gt_quads = _load_gt_quads(val_json_for_run)

    combos = [(t, b) for t in args.thresh for b in args.box_thresh]
    print("=" * 90, flush=True)
    print("Postprocess grid sweep (real CLEval)", flush=True)
    print("=" * 90, flush=True)
    print(f"preset={args.preset}", flush=True)
    print(f"folds={fold_indices}", flush=True)
    print(f"val_json={val_json_for_run}", flush=True)
    print(f"image_dir={args.image_dir}", flush=True)
    print(f"grid: thresh={args.thresh} x box_thresh={args.box_thresh} (total={len(combos)})", flush=True)
    print(f"ensemble: min_votes={args.min_votes}, iou_threshold={args.iou_threshold}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print("=" * 90, flush=True)

    # Write CSV header
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("thresh,box_thresh,precision,recall,hmean,elapsed_sec\n")

    best: SweepResult | None = None

    for (thresh, box_thresh) in tqdm(combos, desc="Grid", position=0):
        start = time.time()
        per_fold_time: Dict[int, float] = {}
        predictions_by_fold: Dict[int, Dict[str, List[List[List[float]]]]] = {}

        print(f"\n[GRID] thresh={thresh:.3f} box_thresh={box_thresh:.3f}", flush=True)

        for fold_idx in fold_indices:
            ckpt = checkpoints[fold_idx]
            t0 = time.time()
            print(f"  - Predict fold {fold_idx}: {ckpt}", flush=True)

            pred_map = _predict_fold_on_val(
                preset=args.preset,
                checkpoint_path=ckpt,
                thresh=thresh,
                box_thresh=box_thresh,
                val_json=val_json_for_run,
                image_dir=args.image_dir,
                devices=args.devices,
                accelerator=args.accelerator,
            )
            predictions_by_fold[fold_idx] = pred_map
            per_fold_time[fold_idx] = time.time() - t0
            print(f"    done: {len(pred_map)} images, {per_fold_time[fold_idx]:.1f}s", flush=True)

        ensembled = _ensemble_improved(
            predictions_by_fold=predictions_by_fold,
            fold_weights=fold_weights,
            min_votes=args.min_votes,
            iou_threshold=args.iou_threshold,
        )

        precision, recall, hmean = _eval_cleval(ensembled, gt_quads)
        elapsed = time.time() - start

        result = SweepResult(
            thresh=float(thresh),
            box_thresh=float(box_thresh),
            precision=precision,
            recall=recall,
            hmean=hmean,
            elapsed_sec=elapsed,
            per_fold_time_sec=per_fold_time,
        )

        if (best is None) or (result.hmean > best.hmean):
            best = result

        print(
            f"  => P={precision:.6f} R={recall:.6f} H={hmean:.6f} (elapsed {elapsed/60:.1f}m)",
            flush=True,
        )
        if best is not None:
            print(
                f"  [BEST] thresh={best.thresh:.3f} box={best.box_thresh:.3f} H={best.hmean:.6f}",
                flush=True,
            )

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "thresh": result.thresh,
                        "box_thresh": result.box_thresh,
                        "precision": result.precision,
                        "recall": result.recall,
                        "hmean": result.hmean,
                        "elapsed_sec": result.elapsed_sec,
                        "per_fold_time_sec": result.per_fold_time_sec,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{result.thresh:.6f},{result.box_thresh:.6f},{result.precision:.8f},{result.recall:.8f},{result.hmean:.8f},{result.elapsed_sec:.2f}\n"
            )

    print("\nDone.", flush=True)
    if best is not None:
        print(
            f"BEST: thresh={best.thresh:.3f} box_thresh={best.box_thresh:.3f} "
            f"P={best.precision:.6f} R={best.recall:.6f} H={best.hmean:.6f}",
            flush=True,
        )
        print(f"Results: {csv_path}", flush=True)
        print(f"Details: {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
