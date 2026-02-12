#!/usr/bin/env python3
"""
최고점 0.9863 기준 TTA + NMS 최종 제출파일 생성
- 기본 파라미터: thresh=0.24, box_thresh=0.27, unclip=2.0 (0.9863 설정)
- TTA: 수평 플립 prob_map 평균
- NMS: 중복 박스 제거
소요시간: ~3분
"""

import os, sys, json, time, shutil
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.models.head.db_postprocess import DBPostProcessor

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
SUBMISSION_DIR = Path("/data/ephemeral/home/baseline_code/outputs/submissions")
OUTPUT_DIR = Path("/data/ephemeral/home/baseline_code/outputs/final_tta_nms")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 0.9863 원본 설정
BEST_THRESH = 0.22
BEST_BOX_THRESH = 0.40
BEST_UNCLIP = 2.0


def polygon_iou(poly1, poly2):
    """AABB 기반 IoU"""
    p1 = np.array(poly1).reshape(-1, 2)
    p2 = np.array(poly2).reshape(-1, 2)
    
    x1_min, y1_min = p1.min(axis=0)
    x1_max, y1_max = p1.max(axis=0)
    x2_min, y2_min = p2.min(axis=0)
    x2_max, y2_max = p2.max(axis=0)
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0.0


def apply_nms(boxes, scores, iou_threshold=0.3):
    """NMS: score 높은 박스 우선 유지"""
    if len(boxes) == 0:
        return boxes, scores
    
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        for j in indices:
            if j in suppressed or j == i:
                continue
            if polygon_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(j)
    
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def export_csv(predictions, name):
    """제출파일 생성"""
    submission = OrderedDict(images=OrderedDict())
    total_boxes = 0
    
    for fname, pred in predictions.items():
        words = OrderedDict()
        for idx, box in enumerate(pred['boxes']):
            words[f'{idx+1:04}'] = OrderedDict(points=box)
        submission['images'][fname] = OrderedDict(words=words)
        total_boxes += len(pred['boxes'])
    
    json_path = OUTPUT_DIR / f"{name}.json"
    with open(json_path, 'w') as f:
        json.dump(submission, f, indent=4)
    
    from ocr.utils.convert_submission import convert_json_to_csv
    csv_path = OUTPUT_DIR / f"{name}.csv"
    convert_json_to_csv(str(json_path), str(csv_path))
    
    final = SUBMISSION_DIR / f"{name}.csv"
    shutil.copy2(csv_path, final)
    
    size_mb = final.stat().st_size / (1024*1024)
    print(f"  ✅ {name}.csv: {total_boxes:,} boxes, {size_mb:.1f}MB")
    return str(final)


def main():
    start = time.time()
    print("="*80)
    print("🎯 최고점(0.9863) 기준 TTA + NMS 최종 제출파일 생성")
    print(f"   thresh={BEST_THRESH}, box_thresh={BEST_BOX_THRESH}, unclip={BEST_UNCLIP}")
    print("="*80)
    
    # 모델 로드
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize(version_base='1.2', config_path='configs'):
        cfg = compose(config_name='predict', overrides=[
            'preset=hrnet_w44_1280',
            f'models.head.postprocess.thresh={BEST_THRESH}',
            f'models.head.postprocess.box_thresh={BEST_BOX_THRESH}',
            f'models.head.postprocess.unclip_ratio={BEST_UNCLIP}',
        ])
        cfg.checkpoint_path = CHECKPOINT
        cfg.minified_json = False
        cfg.submission_dir = str(OUTPUT_DIR / "temp")
        
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
        cfg.disable_predict_export = True
        OmegaConf.set_struct(cfg, True)
        
        model_module, data_module = get_pl_modules_by_cfg(cfg)
    
    device = torch.device('cuda')
    ckpt = torch.load(CHECKPOINT, map_location=device)
    if 'state_dict' in ckpt:
        sd = {k.replace('model.', '', 1): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        model_module.model.load_state_dict(sd)
    model_module = model_module.to(device)
    model_module.eval()
    
    postprocessor = DBPostProcessor(
        thresh=BEST_THRESH, box_thresh=BEST_BOX_THRESH,
        max_candidates=500, use_polygon=True, unclip_ratio=BEST_UNCLIP
    )
    
    predict_loader = data_module.predict_dataloader()
    
    # ======== Phase 1: 기본 + TTA 동시 예측 ========
    print("\n🔄 예측 중 (기본 + TTA 수평플립 동시 처리)...")
    
    preds_normal = OrderedDict()
    preds_tta = OrderedDict()
    
    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="Predict"):
            images = batch['images'].to(device)
            inv_mat = batch['inverse_matrix']
            
            # 원본
            pred_orig = model_module.model(images=images, return_loss=False)
            prob_orig = pred_orig['prob_maps']
            
            # 수평 플립
            pred_flip = model_module.model(images=torch.flip(images, [3]), return_loss=False)
            prob_flip = torch.flip(pred_flip['prob_maps'], [3])
            
            # TTA 평균
            prob_avg = (prob_orig + prob_flip) / 2.0
            
            seg_orig = postprocessor.binarize(prob_orig)
            seg_tta = postprocessor.binarize(prob_avg)
            
            for idx in range(images.size(0)):
                fname = batch['image_filename'][idx]
                
                # 기본
                boxes_n, scores_n = postprocessor.polygons_from_bitmap(
                    prob_orig[idx].cpu(), seg_orig[idx].cpu(), inverse_matrix=inv_mat[idx])
                preds_normal[fname] = {'boxes': boxes_n, 'scores': scores_n}
                
                # TTA
                boxes_t, scores_t = postprocessor.polygons_from_bitmap(
                    prob_avg[idx].cpu(), seg_tta[idx].cpu(), inverse_matrix=inv_mat[idx])
                preds_tta[fname] = {'boxes': boxes_t, 'scores': scores_t}
    
    elapsed_pred = time.time() - start
    print(f"  예측 완료: {elapsed_pred:.0f}초")
    
    # ======== Phase 2: 제출파일 생성 ========
    print("\n📁 제출파일 생성:")
    
    # 1) 기본 (0.9863 재현)
    export_csv(preds_normal, "FINAL_baseline")
    
    # 2) TTA만
    export_csv(preds_tta, "FINAL_tta")
    
    # 3) 기본 + NMS
    for nms_t in [0.2, 0.3]:
        preds_nms = OrderedDict()
        removed = 0
        total = 0
        for fname, pred in preds_normal.items():
            total += len(pred['boxes'])
            b, s = apply_nms(pred['boxes'], pred['scores'], nms_t)
            removed += len(pred['boxes']) - len(b)
            preds_nms[fname] = {'boxes': b, 'scores': s}
        print(f"  NMS(iou={nms_t}): {total:,} → {total-removed:,} ({removed:,} 제거)")
        export_csv(preds_nms, f"FINAL_nms{int(nms_t*10)}")
    
    # 4) TTA + NMS
    for nms_t in [0.2, 0.3]:
        preds_tta_nms = OrderedDict()
        removed = 0
        total = 0
        for fname, pred in preds_tta.items():
            total += len(pred['boxes'])
            b, s = apply_nms(pred['boxes'], pred['scores'], nms_t)
            removed += len(pred['boxes']) - len(b)
            preds_tta_nms[fname] = {'boxes': b, 'scores': s}
        print(f"  TTA+NMS(iou={nms_t}): {total:,} → {total-removed:,} ({removed:,} 제거)")
        export_csv(preds_tta_nms, f"FINAL_tta_nms{int(nms_t*10)}")
    
    total_time = time.time() - start
    
    print("\n" + "="*80)
    print(f"🎉 완료! ({total_time:.0f}초)")
    print("="*80)
    print("\n📁 생성 파일 목록:")
    for f in sorted(SUBMISSION_DIR.glob("FINAL_*.csv")):
        sz = f.stat().st_size / (1024*1024)
        print(f"  {f.name:40s} {sz:.1f}MB")
    
    print("\n💡 제출 추천:")
    print("  1️⃣  FINAL_tta.csv (TTA, 가장 안정적)")
    print("  2️⃣  FINAL_tta_nms3.csv (TTA+NMS, 중복제거)")
    print("  3️⃣  FINAL_baseline.csv (원본 재현)")


if __name__ == "__main__":
    main()
