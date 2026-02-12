#!/usr/bin/env python3
"""
0.215-0.217 범위 미세 탐색
0.218이 local max이므로 바로 아래 범위 탐색
"""
import os
import sys
import json
import lightning.pytorch as pl
from pathlib import Path
from datetime import datetime
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import shutil

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

# 0.218 바로 아래 범위 (Peak 왼쪽)
PARAM_GRID = [
    (0.215, 0.395),  # -0.003
    (0.216, 0.396),  # -0.002
    (0.217, 0.397),  # -0.001
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_fine_tune")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_single_prediction(thresh, box_thresh, idx):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{len(PARAM_GRID)}] thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
    print('='*80)
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    try:
        with initialize(version_base='1.2', config_path='configs'):
            cfg = compose(config_name='predict', overrides=[
                'preset=hrnet_w44_1280',
                f'models.head.postprocess.thresh={thresh}',
                f'models.head.postprocess.box_thresh={box_thresh}',
            ])
            cfg.checkpoint_path = CHECKPOINT
            cfg.minified_json = False
            
            output_name = f"fold3_t{int(thresh*1000)}_b{int(box_thresh*1000)}"
            cfg.submission_dir = str(OUTPUT_BASE / output_name)
            
            from ocr.lightning_modules import get_pl_modules_by_cfg
            model_module, data_module = get_pl_modules_by_cfg(cfg)
            
            trainer = pl.Trainer(logger=False, devices=1)
            trainer.predict(model_module, data_module, ckpt_path=CHECKPOINT)
            
            json_files = list(Path(cfg.submission_dir).glob('*.json'))
            
            if json_files:
                json_file = json_files[0]
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                total_boxes = sum(len(img_data.get('words', {})) for img_data in data['images'].values())
                print(f"✓ 예측 완료: {total_boxes:,} boxes")
                
                csv_file = OUTPUT_BASE / f"{output_name}_fine.csv"
                from ocr.utils.convert_submission import convert_json_to_csv
                result = convert_json_to_csv(str(json_file), str(csv_file))
                
                if result:
                    dest = Path('/data/ephemeral/home/baseline_code/outputs/submissions') / csv_file.name
                    shutil.copy(csv_file, dest)
                    print(f"✓ 제출 폴더 복사: {dest}")
                    
                    return {'thresh': thresh, 'box_thresh': box_thresh, 'boxes': total_boxes, 'csv': str(dest), 'success': True}
                    
    except Exception as e:
        print(f"✗ 에러: {e}")
    
    return {'thresh': thresh, 'box_thresh': box_thresh, 'success': False}

def main():
    print("="*80)
    print("0.218 왼쪽 범위 미세 탐색 (0.215-0.217)")
    print("="*80)
    print("현재까지 결과:")
    print("  0.210/0.390 → 0.9858 (P=0.9882, R=0.9840)")
    print("  0.218/0.398 → 0.9860 (P=0.9888, R=0.9838) ← Peak")
    print("  0.225/0.405 → 0.9855 (P=0.9887, R=0.9829)")
    print()
    print("전략: 0.218이 peak이므로 바로 왼쪽 탐색")
    print()
    
    results = []
    for idx, (thresh, box_thresh) in enumerate(PARAM_GRID):
        result = run_single_prediction(thresh, box_thresh, idx)
        results.append(result)
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
    for r in results:
        if r.get('csv'):
            print(f"  {Path(r['csv']).name}")

if __name__ == "__main__":
    main()
