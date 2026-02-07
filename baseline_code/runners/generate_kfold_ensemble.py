#!/usr/bin/env python3
"""
K-Fold 앙상블 예측 스크립트
각 Fold의 체크포인트로 예측 수행 후 앙상블
"""

import os
import json
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from collections import OrderedDict, defaultdict
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# Hydra 초기화 정리
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# K-Fold 결과 파일 로드
kfold_results_path = "/data/ephemeral/home/baseline_code/hrnet_w44_1280_optimal_kfold_results_20260207_0458.json"
with open(kfold_results_path, 'r') as f:
    kfold_results = json.load(f)

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
    print(f"{'='*80}\n")
    
    # Hydra 초기화 정리 (각 반복마다)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Hydra 설정 로드
    with initialize(version_base='1.2', config_path='../configs'):
        cfg = compose(config_name='predict', overrides=['preset=hrnet_w44_1280'])
        cfg.checkpoint_path = checkpoint_path  # 직접 할당
        cfg.minified_json = False
        
        # 임시 출력 디렉토리 설정
        cfg.submission_dir = f'outputs/kfold_ensemble_temp/fold_{fold_idx}'
        
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

# 앙상블: Bounding Box 투표 방식
print("앙상블 수행 중...")

ensemble_predictions = OrderedDict(images=OrderedDict())

# 모든 이미지에 대해 처리
all_images = set()
for fold_idx, predictions in predictions_by_fold:
    all_images.update(predictions['images'].keys())

print(f"총 {len(all_images)}개 이미지 처리")

for image_name in sorted(all_images):
    # 각 Fold의 예측 결과 수집
    all_boxes_for_image = []
    
    for fold_idx, predictions in predictions_by_fold:
        if image_name in predictions['images']:
            words = predictions['images'][image_name]['words']
            for word_id, word_data in words.items():
                points = word_data['points']
                all_boxes_for_image.append(np.array(points))
    
    # NMS (Non-Maximum Suppression) 유사 방식으로 중복 제거
    # IoU 기반으로 겹치는 박스들을 그룹화하고 평균
    def boxes_iou(box1, box2):
        """두 박스의 IoU 계산 (간단한 근사)"""
        try:
            # 박스의 최소/최대 좌표 계산
            x1_min, y1_min = box1.min(axis=0)
            x1_max, y1_max = box1.max(axis=0)
            x2_min, y2_min = box2.min(axis=0)
            x2_max, y2_max = box2.max(axis=0)
            
            # 교집합 영역
            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)
            
            if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
                return 0.0
            
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            
            # 합집합 영역
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        except:
            return 0.0
    
    # 투표 기반 앙상블 (IoU > 0.5인 박스들을 평균)
    iou_threshold = 0.5
    final_boxes = []
    used = [False] * len(all_boxes_for_image)
    
    for i, box1 in enumerate(all_boxes_for_image):
        if used[i]:
            continue
        
        # 현재 박스와 IoU가 높은 박스들 찾기
        cluster = [box1]
        used[i] = True
        
        for j, box2 in enumerate(all_boxes_for_image):
            if used[j]:
                continue
            if boxes_iou(box1, box2) > iou_threshold:
                cluster.append(box2)
                used[j] = True
        
        # 클러스터의 평균 박스 계산
        if len(cluster) >= 2:  # 2개 이상 Fold에서 감지된 경우만
            avg_box = np.mean(cluster, axis=0)
            final_boxes.append(avg_box)
        elif len(cluster) >= 1:  # 1개 Fold에서만 감지된 경우도 포함 (보수적 접근)
            final_boxes.append(cluster[0])
    
    # 결과 저장
    words = OrderedDict()
    for idx, box in enumerate(final_boxes):
        # 반올림하여 정수 좌표로 변환
        box_int = [[int(round(x)), int(round(y))] for x, y in box]
        words[f'{idx + 1:04}'] = OrderedDict(points=box_int)
    
    ensemble_predictions['images'][image_name] = OrderedDict(words=words)

# 앙상블 결과 저장
ensemble_json_path = Path('outputs/kfold_ensemble_temp/ensemble_result.json')
ensemble_json_path.parent.mkdir(parents=True, exist_ok=True)

with ensemble_json_path.open('w') as f:
    json.dump(ensemble_predictions, f, indent=4)

print(f"✓ 앙상블 결과 저장: {ensemble_json_path}")

# JSON을 CSV로 변환
print("\nCSV 변환 중...")
from ocr.utils.convert_submission import convert_json_to_csv

csv_output_path = '/data/ephemeral/home/hrnet_w44_kfold5_ensemble_submission.csv'
result = convert_json_to_csv(str(ensemble_json_path), csv_output_path)

if result:
    num_rows, output_file = result
    print(f"\n{'='*80}")
    print(f"✓ 앙상블 제출 파일 생성 완료!")
    print(f"파일: {output_file}")
    print(f"이미지 수: {num_rows}")
    print(f"{'='*80}")
else:
    print("\n✗ CSV 변환 실패")

print("\n✓ K-Fold 앙상블 완료!")
