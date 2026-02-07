#!/usr/bin/env python3
"""
개선된 K-Fold 앙상블 예측 스크립트
- 엄격한 투표 임계값 적용 (60% 합의 필요)
- Fold별 가중치 적용 (Val 성능 기반)
- 정교한 폴리곤 IoU 계산
"""

import os
import json
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from collections import OrderedDict, defaultdict
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from shapely.geometry import Polygon
from shapely.validation import make_valid

# Hydra 초기화 정리
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# K-Fold 결과 파일 로드
kfold_results_path = "/data/ephemeral/home/baseline_code/hrnet_w44_1280_optimal_kfold_results_20260207_0458.json"
with open(kfold_results_path, 'r') as f:
    kfold_results = json.load(f)

# Fold별 가중치 (Val 성능 기반)
fold_weights = {
    4: 0.30,  # 0.9837 (최고)
    2: 0.25,  # 0.9781
    3: 0.20,  # 0.9764
    0: 0.15,  # 0.9738
    1: 0.10,  # 0.9717 (최저)
}

print("="*80)
print("개선된 K-Fold 앙상블 예측")
print("="*80)
print("\n앙상블 설정:")
print(f"  - 최소 투표 수: 3개 Fold (60% 합의)")
print(f"  - 가중 앙상블: 활성화")
print(f"  - Fold별 가중치:")
for fold_idx, weight in sorted(fold_weights.items()):
    val_score = kfold_results['fold_results'][f'fold_{fold_idx}']['best_score']
    print(f"    • Fold {fold_idx}: {weight:.2f} (Val={val_score:.4f})")
print()

# 각 Fold의 체크포인트 경로
checkpoints = []
for fold_idx in range(5):
    fold_key = f"fold_{fold_idx}"
    checkpoint_path = kfold_results['fold_results'][fold_key]['best_checkpoint']
    if os.path.exists(checkpoint_path):
        checkpoints.append((fold_idx, checkpoint_path))
        print(f"✓ Fold {fold_idx}: {checkpoint_path}")
    else:
        print(f"✗ Fold {fold_idx}: 체크포인트 없음 - {checkpoint_path}")

print(f"\n총 {len(checkpoints)}개 체크포인트 발견")

# 각 Fold별로 예측 수행
predictions_by_fold = []

for fold_idx, checkpoint_path in checkpoints:
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx} 예측 시작")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"가중치: {fold_weights[fold_idx]:.2f}")
    print(f"{'='*80}\n")
    
    # Hydra 초기화 정리 (각 반복마다)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Hydra 설정 로드
    with initialize(version_base='1.2', config_path='../configs'):
        cfg = compose(config_name='predict', overrides=['preset=hrnet_w44_1280'])
        cfg.checkpoint_path = checkpoint_path
        cfg.minified_json = False
        
        # 임시 출력 디렉토리 설정
        cfg.submission_dir = f'outputs/kfold_ensemble_improved_temp/fold_{fold_idx}'
        
        # 모듈 임포트 및 생성
        import sys
        sys.path.insert(0, '/data/ephemeral/home/baseline_code')
        from ocr.lightning_modules import get_pl_modules_by_cfg
        
        model_module, data_module = get_pl_modules_by_cfg(cfg)
        
        # Trainer 생성 및 예측
        trainer = pl.Trainer(
            logger=False,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=True
        )
        
        print(f"Fold {fold_idx} 예측 중...")
        trainer.predict(model_module, data_module, ckpt_path=checkpoint_path)
        
        # 예측 결과 JSON 파일 찾기
        submission_dir = Path(cfg.submission_dir)
        json_files = sorted(submission_dir.glob("*.json"))
        
        if json_files:
            latest_json = json_files[-1]
            print(f"✓ Fold {fold_idx} 예측 완료: {latest_json}")
            
            # 예측 결과 로드
            with open(latest_json, 'r') as f:
                fold_predictions = json.load(f)
                predictions_by_fold.append((fold_idx, fold_predictions))
        else:
            print(f"✗ Fold {fold_idx} 예측 파일을 찾을 수 없습니다")

print(f"\n{'='*80}")
print(f"모든 Fold 예측 완료! 총 {len(predictions_by_fold)}개 결과")
print(f"{'='*80}\n")

# 개선된 앙상블: 엄격한 투표 + 가중 평균
print("개선된 앙상블 수행 중...")

ensemble_predictions = OrderedDict(images=OrderedDict())

# 모든 이미지에 대해 처리
all_images = set()
for fold_idx, predictions in predictions_by_fold:
    all_images.update(predictions['images'].keys())

print(f"총 {len(all_images)}개 이미지 처리")

# 통계 수집
stats = {
    'total_boxes_per_fold': defaultdict(int),
    'ensemble_boxes': 0,
    'boxes_with_1_vote': 0,
    'boxes_with_2_votes': 0,
    'boxes_with_3_votes': 0,
    'boxes_with_4_votes': 0,
    'boxes_with_5_votes': 0,
}

def polygon_iou(poly1_points, poly2_points):
    """정교한 폴리곤 IoU 계산 (Shapely 사용)"""
    try:
        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)
        
        # 유효하지 않은 폴리곤 수정
        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)
        
        if poly1.is_empty or poly2.is_empty:
            return 0.0
        
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception as e:
        # Shapely 실패 시 간단한 AABB IoU로 폴백
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
            
            return inter_area / union_area if union_area > 0 else 0.0
        except:
            return 0.0

for image_name in sorted(all_images):
    # 각 Fold의 예측 결과 수집 (가중치 포함)
    all_boxes_for_image = []
    
    for fold_idx, predictions in predictions_by_fold:
        if image_name in predictions['images']:
            words = predictions['images'][image_name]['words']
            stats['total_boxes_per_fold'][fold_idx] += len(words)
            
            for word_id, word_data in words.items():
                points = word_data['points']
                weight = fold_weights[fold_idx]
                all_boxes_for_image.append({
                    'points': np.array(points),
                    'fold_idx': fold_idx,
                    'weight': weight
                })
    
    # 엄격한 투표 기반 앙상블 (IoU > 0.5, 최소 3개 Fold 합의)
    iou_threshold = 0.5
    min_votes = 3  # Reverted from 2 (precision recovery)  # 5개 중 3개 이상 (60% 합의)
    
    final_boxes = []
    used = [False] * len(all_boxes_for_image)
    
    for i, box1 in enumerate(all_boxes_for_image):
        if used[i]:
            continue
        
        # 현재 박스와 IoU가 높은 박스들 찾기
        cluster = [box1]
        cluster_fold_indices = {box1['fold_idx']}
        used[i] = True
        
        for j, box2 in enumerate(all_boxes_for_image):
            if used[j]:
                continue
            
            # 같은 Fold에서 나온 박스는 제외 (중복 방지)
            if box2['fold_idx'] in cluster_fold_indices:
                continue
            
            iou = polygon_iou(box1['points'], box2['points'])
            if iou > iou_threshold:
                cluster.append(box2)
                cluster_fold_indices.add(box2['fold_idx'])
                used[j] = True
        
        # 통계 수집
        vote_count = len(cluster)
        if vote_count == 1:
            stats['boxes_with_1_vote'] += 1
        elif vote_count == 2:
            stats['boxes_with_2_votes'] += 1
        elif vote_count == 3:
            stats['boxes_with_3_votes'] += 1
        elif vote_count == 4:
            stats['boxes_with_4_votes'] += 1
        elif vote_count == 5:
            stats['boxes_with_5_votes'] += 1
        
        # ✓ 최소 3개 Fold 이상에서 감지된 경우만 포함
        if len(cluster) >= min_votes:
            # POLY 모드: 점 개수가 다르므로 가중 평균 불가
            # → 가장 높은 가중치를 가진 박스 선택
            best_box = max(cluster, key=lambda b: b['weight'])
            final_boxes.append(best_box['points'])
            stats['ensemble_boxes'] += 1
    
    # 결과 저장
    words = OrderedDict()
    for idx, box in enumerate(final_boxes):
        # 반올림하여 정수 좌표로 변환
        box_int = [[int(round(x)), int(round(y))] for x, y in box]
        words[f'{idx + 1:04}'] = OrderedDict(points=box_int)
    
    ensemble_predictions['images'][image_name] = OrderedDict(words=words)

# 통계 출력
print(f"\n{'='*80}")
print("앙상블 통계")
print(f"{'='*80}")
print(f"\nFold별 원본 박스 수:")
for fold_idx in sorted(stats['total_boxes_per_fold'].keys()):
    count = stats['total_boxes_per_fold'][fold_idx]
    print(f"  Fold {fold_idx}: {count:,}개")

total_original = sum(stats['total_boxes_per_fold'].values())
print(f"  총합: {total_original:,}개")

print(f"\n투표 분포:")
print(f"  1개 Fold만 감지: {stats['boxes_with_1_vote']:,}개 (제외됨 ✗)")
print(f"  2개 Fold 합의: {stats['boxes_with_2_votes']:,}개 (제외됨 ✗)")
print(f"  3개 Fold 합의: {stats['boxes_with_3_votes']:,}개 (포함 ✓)")
print(f"  4개 Fold 합의: {stats['boxes_with_4_votes']:,}개 (포함 ✓)")
print(f"  5개 Fold 합의: {stats['boxes_with_5_votes']:,}개 (포함 ✓)")

print(f"\n최종 앙상블 결과:")
print(f"  포함된 박스: {stats['ensemble_boxes']:,}개")
excluded = total_original - stats['ensemble_boxes']
exclusion_rate = (excluded / total_original * 100) if total_original > 0 else 0
print(f"  제외된 박스: {excluded:,}개 ({exclusion_rate:.1f}%)")

avg_boxes_per_image = stats['ensemble_boxes'] / len(all_images)
print(f"  평균 박스/이미지: {avg_boxes_per_image:.1f}개")

# 앙상블 결과 저장
ensemble_json_path = Path('outputs/kfold_ensemble_improved_temp/ensemble_result.json')
ensemble_json_path.parent.mkdir(parents=True, exist_ok=True)

with ensemble_json_path.open('w') as f:
    json.dump(ensemble_predictions, f, indent=4)

print(f"\n✓ 앙상블 결과 저장: {ensemble_json_path}")

# JSON을 CSV로 변환
print("\nCSV 변환 중...")
from ocr.utils.convert_submission import convert_json_to_csv

csv_output_path = '/data/ephemeral/home/hrnet_w44_kfold5_ensemble_improved.csv'
result = convert_json_to_csv(str(ensemble_json_path), csv_output_path)

if result:
    num_rows, output_file = result
    print(f"\n{'='*80}")
    print(f"✓ 개선된 앙상블 제출 파일 생성 완료!")
    print(f"{'='*80}")
    print(f"파일: {output_file}")
    print(f"이미지 수: {num_rows}")
    print(f"\n예상 성능:")
    print(f"  Hmean: 0.974 ~ 0.977 (이전 0.9421 대비 개선)")
    print(f"  Precision: 0.973+ (이전 0.9273 대비 개선)")
    print(f"  Recall: 0.970+ (유사 유지)")
    print(f"{'='*80}")
else:
    print("\n✗ CSV 변환 실패")

print("\n✓ 개선된 K-Fold 앙상블 완료!")
