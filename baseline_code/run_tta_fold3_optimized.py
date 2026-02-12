#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) for Fold 3
원본 + HFlip + 밝기 조정 앙상블
"""
import os
import sys
import json
import shutil
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from collections import OrderedDict
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"

# TTA 조합 (각각 별도 예측 후 병합)
TTA_CONFIGS = [
    {'name': 'original', 'thresh': 0.218, 'box_thresh': 0.398},
    {'name': 'thresh_low', 'thresh': 0.215, 'box_thresh': 0.395},  # 낮은 threshold
    {'name': 'thresh_high', 'thresh': 0.221, 'box_thresh': 0.401}, # 높은 threshold
]

OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_tta")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_tta_prediction(config):
    """단일 TTA 설정으로 예측"""
    print(f"\n{'='*60}")
    print(f"TTA: {config['name']}")
    print(f"  thresh={config['thresh']:.3f}, box_thresh={config['box_thresh']:.3f}")
    print('='*60)
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    try:
        with initialize(version_base='1.2', config_path='configs'):
            cfg = compose(config_name='predict', overrides=[
                'preset=hrnet_w44_1280',
                f'models.head.postprocess.thresh={config["thresh"]}',
                f'models.head.postprocess.box_thresh={config["box_thresh"]}',
            ])
            cfg.checkpoint_path = CHECKPOINT
            cfg.minified_json = False
            cfg.submission_dir = str(OUTPUT_BASE / config['name'])
            
            from ocr.lightning_modules import get_pl_modules_by_cfg
            model_module, data_module = get_pl_modules_by_cfg(cfg)
            
            trainer = pl.Trainer(logger=False, devices=1)
            trainer.predict(model_module, data_module, ckpt_path=CHECKPOINT)
            
            json_file = list(Path(cfg.submission_dir).glob('*.json'))[0]
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            total_boxes = sum(len(img_data.get('words', {})) for img_data in data['images'].values())
            print(f"✓ 예측 완료: {total_boxes:,} boxes")
            
            return json_file, data
            
    except Exception as e:
        print(f"✗ 에러: {e}")
        return None, None

def merge_tta_predictions(predictions_list):
    """TTA 예측 병합: 중복 제거 + voting"""
    print("\n" + "="*80)
    print("TTA 병합 중...")
    print("="*80)
    
    # 모든 이미지 리스트
    all_images = set()
    for pred_data in predictions_list:
        if pred_data:
            all_images.update(pred_data['images'].keys())
    
    merged = OrderedDict(images=OrderedDict())
    
    for img_name in tqdm(sorted(all_images), desc="TTA 앙상블"):
        # 각 TTA에서 해당 이미지의 박스 수집
        all_boxes = []
        
        for pred_data in predictions_list:
            if pred_data and img_name in pred_data['images']:
                words = pred_data['images'][img_name].get('words', {})
                for word_info in words.values():
                    all_boxes.append(word_info)
        
        # 중복 제거 (좌표 기반)
        unique_boxes = {}
        for box in all_boxes:
            # 박스를 튜플로 변환해서 키로 사용
            points_key = tuple(tuple(p) for p in box['points'])
            
            if points_key not in unique_boxes:
                unique_boxes[points_key] = box
            # 이미 있으면 스킵 (중복 제거)
        
        # 결과 저장
        merged['images'][img_name] = OrderedDict(words=OrderedDict())
        for idx, box in enumerate(unique_boxes.values()):
            merged['images'][img_name]['words'][str(idx)] = box
    
    total_boxes = sum(len(img['words']) for img in merged['images'].values())
    print(f"✓ 병합 완료: {total_boxes:,} boxes (중복 제거됨)")
    
    return merged

def main():
    print("="*80)
    print("Test-Time Augmentation (TTA) - Fold 3")
    print("="*80)
    print(f"TTA 조합: {len(TTA_CONFIGS)}개")
    print("전략: Multi-threshold 앙상블 (중복 제거)")
    print()
    
    # 각 TTA 예측 실행
    predictions = []
    for config in TTA_CONFIGS:
        json_file, pred_data = run_tta_prediction(config)
        if pred_data:
            predictions.append(pred_data)
    
    if len(predictions) < 2:
        print("❌ TTA 예측 실패 (최소 2개 필요)")
        return
    
    # TTA 병합
    merged_data = merge_tta_predictions(predictions)
    
    # JSON 저장
    output_json = OUTPUT_BASE / "fold3_tta_merged.json"
    with open(output_json, 'w') as f:
        json.dump(merged_data, f)
    
    print(f"\n✓ TTA JSON 저장: {output_json}")
    
    # CSV 변환
    csv_file = OUTPUT_BASE / "fold3_tta_merged.csv"
    from ocr.utils.convert_submission import convert_json_to_csv
    result = convert_json_to_csv(str(output_json), str(csv_file))
    
    if result:
        file_size_mb = csv_file.stat().st_size / (1024*1024)
        print(f"✓ CSV 생성: {csv_file.name} ({file_size_mb:.1f}MB)")
        
        # 제출 폴더로 복사
        dest = Path('/data/ephemeral/home/baseline_code/outputs/submissions') / csv_file.name
        shutil.copy(csv_file, dest)
        print(f"✓ 제출 파일: {dest}")
        
        print("\n" + "="*80)
        print("✅ TTA 완료!")
        print("="*80)
        print(f"제출 파일: {dest.name}")
        print("예상 효과: Multi-threshold로 Precision/Recall 균형 개선")

if __name__ == "__main__":
    main()
