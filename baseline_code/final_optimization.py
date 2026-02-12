#!/usr/bin/env python3
"""
=================================================================
Phase 1: Fold 3 Val 로컬 평가 (파라미터 정밀 탐색)
Phase 2: 최적 파라미터 + TTA + NMS → 최종 제출파일 생성
=================================================================
1단계: 모델 Forward Pass 1회 → prob_maps 캐시
2단계: thresh/box_thresh/unclip 조합 sweep → CLEval 로컬 평가
3단계: 최적 파라미터로 test 예측 + TTA + NMS → 최종 제출
"""

import os, sys, json, time, shutil
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

import lightning.pytorch as pl
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.models.head.db_postprocess import DBPostProcessor
from ocr.metrics import CLEvalMetric

# ======================================================================
# Configuration
# ======================================================================
CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
FOLD3_VAL_JSON = "/data/ephemeral/home/data/datasets/jsons/kfold/fold3_val.json"
IMAGE_PATH = "/data/ephemeral/home/data/datasets/images/all"
OUTPUT_DIR = Path("/data/ephemeral/home/baseline_code/outputs/final_optimization")
SUBMISSION_DIR = Path("/data/ephemeral/home/baseline_code/outputs/submissions")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)


def load_model_and_data():
    """모델과 데이터 로드"""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize(version_base='1.2', config_path='configs'):
        cfg = compose(config_name='predict', overrides=[
            'preset=hrnet_w44_1280',
            # Fold 3 val 데이터로 오버라이드
            f'datasets.val_dataset.annotation_path={FOLD3_VAL_JSON}',
            f'datasets.test_dataset.annotation_path={FOLD3_VAL_JSON}',
        ])
        cfg.checkpoint_path = CHECKPOINT
        cfg.minified_json = False
        cfg.submission_dir = str(OUTPUT_DIR / "temp")
        
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
        cfg.disable_predict_export = True
        OmegaConf.set_struct(cfg, True)
        
        model_module, data_module = get_pl_modules_by_cfg(cfg)
    
    return model_module, data_module, cfg


def load_gt_annotations():
    """Fold 3 val GT 로드"""
    with open(FOLD3_VAL_JSON, 'r') as f:
        annotations = json.load(f)
    
    gt_anns = OrderedDict()
    for filename, img_data in annotations['images'].items():
        if 'words' in img_data:
            polygons = [np.array([np.round(word_data['points'])], dtype=np.int32)
                       for word_data in img_data['words'].values()
                       if len(word_data['points'])]
            gt_anns[filename] = polygons
    
    print(f"  GT 로드 완료: {len(gt_anns)} 이미지")
    return gt_anns


# ======================================================================
# Phase 1: Val 로컬 평가 (prob_maps 캐시 + 파라미터 sweep)
# ======================================================================

def cache_prob_maps(model_module, data_module, cfg):
    """모델 Forward Pass 1회 → prob_maps 캐시"""
    print("\n" + "="*80)
    print("Phase 1-A: 모델 Forward Pass → prob_maps 캐시")
    print("="*80)
    
    # val 데이터로더 사용
    model_module.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 체크포인트 로드
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 'model.' prefix 제거
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model_module.model.load_state_dict(new_state_dict)
    
    model_module = model_module.to(device)
    model_module.eval()
    
    # Val dataloader
    val_loader = data_module.val_dataloader()
    
    cached_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Caching prob_maps")):
            # Move to device
            images = batch['images'].to(device)
            
            # Forward pass
            pred = model_module.model(images=images, return_loss=False)
            
            # Cache: prob_maps + batch metadata
            prob_maps = pred['prob_maps'].cpu()
            
            for idx in range(images.size(0)):
                cached_data.append({
                    'filename': batch['image_filename'][idx],
                    'prob_map': prob_maps[idx],  # (1, H, W)
                    'inverse_matrix': batch['inverse_matrix'][idx],
                    'images_shape': images[idx].shape,  # for batch reconstruction
                })
    
    print(f"  캐시 완료: {len(cached_data)} 이미지")
    return cached_data


def evaluate_params(cached_data, gt_anns, thresh, box_thresh, unclip_ratio, max_candidates=500):
    """특정 파라미터 조합으로 CLEval 평가"""
    postprocessor = DBPostProcessor(
        thresh=thresh,
        box_thresh=box_thresh,
        max_candidates=max_candidates,
        use_polygon=True,
        unclip_ratio=unclip_ratio
    )
    
    metric = CLEvalMetric()
    cleval_metrics = defaultdict(list)
    
    for item in cached_data:
        filename = item['filename']
        prob_map = item['prob_map']
        inverse_matrix = item['inverse_matrix']
        
        if filename not in gt_anns:
            continue
        
        # Binarize
        segmentation = postprocessor.binarize(prob_map)
        
        # Get polygons
        boxes, scores = postprocessor.polygons_from_bitmap(
            prob_map, segmentation, inverse_matrix=inverse_matrix
        )
        
        # Compute CLEval
        gt_words = gt_anns[filename]
        det_quads = [[point for coord in polygons for point in coord] for polygons in boxes]
        gt_quads = [item_gt.squeeze().reshape(-1) for item_gt in gt_words]
        
        metric(det_quads, gt_quads)
        cleval = metric.compute()
        cleval_metrics['recall'].append(cleval['det_r'].cpu().numpy())
        cleval_metrics['precision'].append(cleval['det_p'].cpu().numpy())
        cleval_metrics['hmean'].append(cleval['det_h'].cpu().numpy())
        metric.reset()
    
    recall = np.mean(cleval_metrics['recall'])
    precision = np.mean(cleval_metrics['precision'])
    hmean = np.mean(cleval_metrics['hmean'])
    
    return {
        'thresh': thresh,
        'box_thresh': box_thresh,
        'unclip_ratio': unclip_ratio,
        'precision': precision,
        'recall': recall,
        'hmean': hmean
    }


def run_parameter_sweep(cached_data, gt_anns):
    """파라미터 정밀 탐색"""
    print("\n" + "="*80)
    print("Phase 1-B: 파라미터 정밀 탐색 (로컬 Val 평가)")
    print("="*80)
    
    # 탐색 범위 정의
    # thresh: 0.210 ~ 0.222 (0.001 간격) = 13개
    # box_thresh: thresh + [0.176, 0.178, 0.180, 0.182, 0.184] = 5개
    # unclip_ratio: [1.9, 1.95, 2.0] = 3개
    # 총 13 * 5 * 3 = 195개 조합
    
    thresh_values = [round(0.210 + i * 0.001, 3) for i in range(13)]  # 0.210 ~ 0.222
    bt_offsets = [0.176, 0.178, 0.180, 0.182, 0.184]
    unclip_values = [1.9, 1.95, 2.0]
    
    total = len(thresh_values) * len(bt_offsets) * len(unclip_values)
    print(f"  탐색 조합 수: {total}")
    print(f"  thresh: {thresh_values}")
    print(f"  box_thresh offsets: {bt_offsets}")
    print(f"  unclip_ratio: {unclip_values}")
    print()
    
    results = []
    best_hmean = 0
    best_result = None
    
    pbar = tqdm(total=total, desc="Parameter Sweep")
    
    for thresh in thresh_values:
        for bt_offset in bt_offsets:
            box_thresh = round(thresh + bt_offset, 3)
            for unclip in unclip_values:
                result = evaluate_params(cached_data, gt_anns, thresh, box_thresh, unclip)
                results.append(result)
                
                if result['hmean'] > best_hmean:
                    best_hmean = result['hmean']
                    best_result = result
                    pbar.set_postfix({
                        'best_H': f"{best_hmean:.4f}",
                        't': f"{result['thresh']:.3f}",
                        'bt': f"{result['box_thresh']:.3f}",
                        'u': f"{result['unclip_ratio']:.2f}"
                    })
                
                pbar.update(1)
    
    pbar.close()
    
    # 결과 정렬
    results.sort(key=lambda x: x['hmean'], reverse=True)
    
    # Top 20 출력
    print("\n" + "="*80)
    print("📊 Top 20 파라미터 조합 (Val H-Mean 순)")
    print("="*80)
    print(f"{'순위':>4} │{'thresh':>7}│{'box_th':>7}│{'unclip':>6}│{'P':>7}│{'R':>7}│{'H-Mean':>7}")
    print("─" * 55)
    
    for idx, r in enumerate(results[:20], 1):
        marker = "⭐" if idx == 1 else f"{idx:2d}"
        print(f" {marker}  │{r['thresh']:>7.3f}│{r['box_thresh']:>7.3f}│{r['unclip_ratio']:>6.2f}│"
              f"{r['precision']:>7.4f}│{r['recall']:>7.4f}│{r['hmean']:>7.4f}")
    
    print()
    print(f"🏆 최적: thresh={best_result['thresh']:.3f}, box_thresh={best_result['box_thresh']:.3f}, "
          f"unclip={best_result['unclip_ratio']:.2f}")
    print(f"   P={best_result['precision']:.4f}, R={best_result['recall']:.4f}, "
          f"H={best_result['hmean']:.4f}")
    
    return results, best_result


# ======================================================================
# Phase 2: TTA + NMS
# ======================================================================

def polygon_iou(poly1, poly2):
    """두 폴리곤 간 IoU 계산 (AABB 기반)"""
    # AABB (Axis-Aligned Bounding Box) IoU
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
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def apply_nms(boxes, scores, iou_threshold=0.3):
    """NMS 적용 - 겹치는 박스 중 score 높은 것만 유지"""
    if len(boxes) == 0:
        return boxes, scores
    
    # score 기준 내림차순 정렬
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i in sorted_indices:
        if i in suppressed:
            continue
        keep.append(i)
        
        for j in sorted_indices:
            if j in suppressed or j == i:
                continue
            iou = polygon_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                suppressed.add(j)
    
    new_boxes = [boxes[i] for i in keep]
    new_scores = [scores[i] for i in keep]
    
    return new_boxes, new_scores


def tta_predict_with_flip(model_module, data_module, postprocessor, device):
    """TTA: 원본 + 수평 플립 예측 합성"""
    print("\n" + "="*80)
    print("Phase 2-A: TTA 예측 (원본 + 수평 플립 → prob_map 평균)")
    print("="*80)
    
    predict_loader = data_module.predict_dataloader()
    
    all_predictions = OrderedDict()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(predict_loader, desc="TTA Prediction")):
            images = batch['images'].to(device)
            inverse_matrix = batch['inverse_matrix']
            
            # 1) 원본 예측
            pred_orig = model_module.model(images=images, return_loss=False)
            prob_orig = pred_orig['prob_maps']  # (B, 1, H, W)
            
            # 2) 수평 플립 예측
            images_flipped = torch.flip(images, dims=[3])  # W 축 플립
            pred_flip = model_module.model(images=images_flipped, return_loss=False)
            prob_flip = torch.flip(pred_flip['prob_maps'], dims=[3])  # 다시 플립
            
            # 3) prob_map 평균 (Soft TTA)
            prob_avg = (prob_orig + prob_flip) / 2.0
            
            # 4) 폴리곤 추출 (averaged prob_map)
            segmentation = postprocessor.binarize(prob_avg)
            
            for idx in range(images.size(0)):
                filename = batch['image_filename'][idx]
                
                boxes, scores = postprocessor.polygons_from_bitmap(
                    prob_avg[idx].cpu(),
                    segmentation[idx].cpu(),
                    inverse_matrix=inverse_matrix[idx]
                )
                
                all_predictions[filename] = {
                    'boxes': boxes,
                    'scores': scores
                }
    
    print(f"  TTA 예측 완료: {len(all_predictions)} 이미지")
    return all_predictions


def predict_normal(model_module, data_module, postprocessor, device):
    """기본 예측 (TTA 없이)"""
    print("\n" + "="*80)
    print("Phase 2-A: 일반 예측 (TTA 없이)")
    print("="*80)
    
    predict_loader = data_module.predict_dataloader()
    all_predictions = OrderedDict()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(predict_loader, desc="Prediction")):
            images = batch['images'].to(device)
            inverse_matrix = batch['inverse_matrix']
            
            pred = model_module.model(images=images, return_loss=False)
            prob_maps = pred['prob_maps']
            segmentation = postprocessor.binarize(prob_maps)
            
            for idx in range(images.size(0)):
                filename = batch['image_filename'][idx]
                
                boxes, scores = postprocessor.polygons_from_bitmap(
                    prob_maps[idx].cpu(),
                    segmentation[idx].cpu(),
                    inverse_matrix=inverse_matrix[idx]
                )
                
                all_predictions[filename] = {
                    'boxes': boxes,
                    'scores': scores
                }
    
    print(f"  예측 완료: {len(all_predictions)} 이미지")
    return all_predictions


def apply_nms_to_predictions(predictions, nms_threshold=0.3):
    """전체 예측에 NMS 적용"""
    print("\n" + "="*80)
    print(f"Phase 2-B: NMS 적용 (IoU threshold={nms_threshold})")
    print("="*80)
    
    total_before = 0
    total_after = 0
    
    nms_predictions = OrderedDict()
    
    for filename, pred in tqdm(predictions.items(), desc="NMS"):
        boxes = pred['boxes']
        scores = pred['scores']
        
        total_before += len(boxes)
        
        nms_boxes, nms_scores = apply_nms(boxes, scores, nms_threshold)
        
        total_after += len(nms_boxes)
        
        nms_predictions[filename] = {
            'boxes': nms_boxes,
            'scores': nms_scores
        }
    
    removed = total_before - total_after
    print(f"  NMS 결과: {total_before:,} → {total_after:,} 박스 ({removed:,} 제거, {removed/max(total_before,1)*100:.1f}%)")
    
    return nms_predictions


def export_submission(predictions, output_name, best_params):
    """제출파일 생성"""
    print(f"\n  📁 제출파일 생성: {output_name}")
    
    # JSON 생성
    submission = OrderedDict(images=OrderedDict())
    total_boxes = 0
    
    for filename, pred in predictions.items():
        boxes = OrderedDict()
        for idx, box in enumerate(pred['boxes']):
            boxes[f'{idx + 1:04}'] = OrderedDict(points=box)
        submission['images'][filename] = OrderedDict(words=boxes)
        total_boxes += len(pred['boxes'])
    
    # JSON 저장
    json_path = OUTPUT_DIR / f"{output_name}.json"
    with open(json_path, 'w') as f:
        json.dump(submission, f, indent=4)
    
    # CSV 변환
    from ocr.utils.convert_submission import convert_json_to_csv
    csv_path = OUTPUT_DIR / f"{output_name}.csv"
    convert_json_to_csv(str(json_path), str(csv_path))
    
    # submissions 폴더로 복사
    final_path = SUBMISSION_DIR / f"{output_name}.csv"
    shutil.copy2(csv_path, final_path)
    
    file_size = final_path.stat().st_size / (1024*1024)
    print(f"  ✅ {final_path.name}: {total_boxes:,} boxes, {file_size:.1f}MB")
    
    return str(final_path)


# ======================================================================
# Main
# ======================================================================

def main():
    start_time = time.time()
    
    print("="*80)
    print("🎯 최종 최적화: Val 정밀 탐색 → TTA + NMS → 제출파일 생성")
    print("="*80)
    
    # ---- Phase 1: 로컬 Val 평가 ----
    print("\n📦 모델 및 데이터 로드...")
    model_module, data_module, cfg = load_model_and_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 체크포인트 로드
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model_module.model.load_state_dict(new_state_dict)
    
    model_module = model_module.to(device)
    model_module.eval()
    
    # GT 로드
    gt_anns = load_gt_annotations()
    
    # prob_maps 캐시
    cached_data = cache_prob_maps(model_module, data_module, cfg)
    
    # 파라미터 sweep
    all_results, best_result = run_parameter_sweep(cached_data, gt_anns)
    
    # 결과 저장
    results_path = OUTPUT_DIR / "val_sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump([{k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in r.items()} for r in all_results], f, indent=2)
    print(f"\n  결과 저장: {results_path}")
    
    # cached_data에서 prob_maps 메모리 해제 (GPU 메모리 확보)
    del cached_data
    torch.cuda.empty_cache()
    
    best_thresh = best_result['thresh']
    best_bt = best_result['box_thresh']
    best_unclip = best_result['unclip_ratio']
    
    print(f"\n🏆 Val 최적: thresh={best_thresh:.3f}, bt={best_bt:.3f}, unclip={best_unclip:.2f}")
    print(f"   Val H-Mean={best_result['hmean']:.4f}, P={best_result['precision']:.4f}, R={best_result['recall']:.4f}")
    
    # ---- Phase 2: 최적 파라미터로 Test 예측 + TTA + NMS ----
    print("\n" + "="*80)
    print("Phase 2: Test 예측 (최적 파라미터 + TTA + NMS)")
    print("="*80)
    
    postprocessor = DBPostProcessor(
        thresh=best_thresh,
        box_thresh=best_bt,
        max_candidates=500,
        use_polygon=True,
        unclip_ratio=best_unclip
    )
    
    # 2-1: 기본 예측 (비교용)
    predictions_normal = predict_normal(model_module, data_module, postprocessor, device)
    
    t_str = f"{int(best_thresh*1000):03d}"
    bt_str = f"{int(best_bt*1000):03d}"
    u_str = f"{int(best_unclip*100):03d}"
    
    path_normal = export_submission(
        predictions_normal,
        f"FINAL_t{t_str}_b{bt_str}_u{u_str}_normal",
        best_result
    )
    
    # 2-2: TTA 예측
    predictions_tta = tta_predict_with_flip(model_module, data_module, postprocessor, device)
    path_tta = export_submission(
        predictions_tta,
        f"FINAL_t{t_str}_b{bt_str}_u{u_str}_tta",
        best_result
    )
    
    # 2-3: TTA + NMS
    for nms_thresh in [0.2, 0.3, 0.4]:
        predictions_tta_nms = apply_nms_to_predictions(
            deepcopy(predictions_tta), nms_threshold=nms_thresh
        )
        path_tta_nms = export_submission(
            predictions_tta_nms,
            f"FINAL_t{t_str}_b{bt_str}_u{u_str}_tta_nms{int(nms_thresh*10)}",
            best_result
        )
    
    # 2-4: Normal + NMS (NMS만 적용)
    for nms_thresh in [0.2, 0.3]:
        predictions_nms = apply_nms_to_predictions(
            deepcopy(predictions_normal), nms_threshold=nms_thresh
        )
        path_nms = export_submission(
            predictions_nms,
            f"FINAL_t{t_str}_b{bt_str}_u{u_str}_nms{int(nms_thresh*10)}",
            best_result
        )
    
    total_time = time.time() - start_time
    
    # ---- 최종 요약 ----
    print("\n" + "="*80)
    print("🎉 최종 완료!")
    print("="*80)
    print(f"\n총 소요시간: {total_time/60:.1f}분")
    print()
    print(f"🏆 Val 최적 파라미터:")
    print(f"   thresh={best_thresh:.3f}")
    print(f"   box_thresh={best_bt:.3f}")
    print(f"   unclip_ratio={best_unclip:.2f}")
    print(f"   Val H-Mean={best_result['hmean']:.4f}")
    print()
    print("📁 생성된 제출 파일:")
    
    final_files = sorted(SUBMISSION_DIR.glob("FINAL_*.csv"))
    for f in final_files:
        size = f.stat().st_size / (1024*1024)
        print(f"   {f.name} ({size:.1f}MB)")
    
    print()
    print("💡 제출 권장 순서:")
    print("   1. FINAL_*_tta.csv (TTA만, 가장 안전)")
    print("   2. FINAL_*_tta_nms3.csv (TTA+NMS, 정밀도 개선 기대)")
    print("   3. FINAL_*_normal.csv (기본, 비교용)")
    print()


if __name__ == "__main__":
    main()
