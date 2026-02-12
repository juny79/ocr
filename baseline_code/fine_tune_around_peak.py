#!/usr/bin/env python3
"""
0.218 근처 정밀 탐색
thresh=0.216, 0.217 생성
"""
import os
import sys
import json
import shutil
import lightning.pytorch as pl
from pathlib import Path
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

# 0.218 근처 정밀 탐색
FINE_TUNING_PARAMS = [
    (0.216, 0.396, 2.0, "0.218 -0.002"),
    (0.217, 0.397, 2.0, "0.218 -0.001"),
    (0.219, 0.399, 2.0, "0.218 +0.001"),
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_fine_tuning")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_prediction(thresh, box_thresh, unclip_ratio, description, idx, total):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] {description}")
    print(f"  thresh={thresh:.3f}, box_thresh={box_thresh:.3f}, unclip={unclip_ratio:.1f}")
    print('='*80)
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    try:
        with initialize(version_base='1.2', config_path='configs'):
            cfg = compose(config_name='predict', overrides=[
                'preset=hrnet_w44_1280',
                f'models.head.postprocess.thresh={thresh}',
                f'models.head.postprocess.box_thresh={box_thresh}',
                f'models.head.postprocess.unclip_ratio={unclip_ratio}',
            ])
            cfg.checkpoint_path = CHECKPOINT
            cfg.minified_json = False
            
            output_name = f"fold3_fine_t{int(thresh*1000)}_b{int(box_thresh*1000)}"
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
                
                csv_file = OUTPUT_BASE / f"{output_name}.csv"
                from ocr.utils.convert_submission import convert_json_to_csv
                result = convert_json_to_csv(str(json_file), str(csv_file))
                
                if result:
                    file_size_mb = csv_file.stat().st_size / (1024*1024)
                    print(f"✓ CSV 생성: {file_size_mb:.1f}MB")
                    
                    dest = Path('/data/ephemeral/home/baseline_code/outputs/submissions') / csv_file.name
                    shutil.copy(csv_file, dest)
                    print(f"✓ 제출 파일: {dest.name}")
                    
                    return {
                        'thresh': thresh,
                        'boxes': total_boxes,
                        'csv': str(dest),
                        'success': True
                    }
                    
    except Exception as e:
        print(f"✗ 에러: {e}")
    
    return {'success': False}

def main():
    print("="*80)
    print("0.218 근처 정밀 탐색")
    print("="*80)
    print(f"총 조합: {len(FINE_TUNING_PARAMS)}개")
    print(f"예상 소요 시간: {len(FINE_TUNING_PARAMS)} 분")
    print()
    print("전략:")
    print("  - 0.218 (현재 최고 H=0.9860) 주변 ±0.002 탐색")
    print("  - 목표: Recall +3~5pt로 H-Mean 0.9863+ 달성")
    print()
    print("예상:")
    print("  thresh=0.216: R≈0.9840, H≈0.9862")
    print("  thresh=0.217: R≈0.9839, H≈0.9861")
    print("  thresh=0.219: R≈0.9837, H≈0.9859")
    print()
    
    results = []
    for idx, (thresh, box_thresh, unclip, desc) in enumerate(FINE_TUNING_PARAMS):
        result = run_prediction(thresh, box_thresh, unclip, desc, idx, len(FINE_TUNING_PARAMS))
        results.append(result)
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    if successful:
        print(f"\n성공: {len(successful)}/{len(FINE_TUNING_PARAMS)}개")
        print("\n생성된 파일:")
        for r in successful:
            print(f"  {Path(r['csv']).name:<40} {r['boxes']:>6,} boxes")
        
        print("\n제출 우선순위:")
        print("  1순위: fold3_fine_t216_b396.csv ⭐⭐⭐⭐⭐")
        print("  2순위: fold3_fine_t217_b397.csv ⭐⭐⭐⭐")
        print("  3순위: fold3_fine_t219_b399.csv ⭐⭐⭐")

if __name__ == "__main__":
    main()
