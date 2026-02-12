#!/usr/bin/env python3
"""
Max Candidates 조정 실험
더 많은 텍스트 영역 후보 검출로 Recall 향상
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

# Max candidates + 최적 thresh/box_thresh 조합
PARAM_GRID = [
    (0.218, 0.398, 700),
    (0.218, 0.398, 1000),
    (0.218, 0.398, 1500),
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_maxcand")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_single_prediction(thresh, box_thresh, max_cand, idx, total):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] max_candidates={max_cand}")
    print(f"  thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
    print('='*80)
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    try:
        with initialize(version_base='1.2', config_path='configs'):
            cfg = compose(config_name='predict', overrides=[
                'preset=hrnet_w44_1280',
                f'models.head.postprocess.thresh={thresh}',
                f'models.head.postprocess.box_thresh={box_thresh}',
                f'models.head.postprocess.max_candidates={max_cand}',
            ])
            cfg.checkpoint_path = CHECKPOINT
            cfg.minified_json = False
            
            output_name = f"fold3_maxcand{max_cand}_t{int(thresh*1000)}_b{int(box_thresh*1000)}"
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
                    
                    dest = Path('/data/ephemeral/home/baseline_code/outputs/submissions') / csv_file.name
                    shutil.copy(csv_file, dest)
                    print(f"✓ 제출 파일 생성: {dest.name} ({file_size_mb:.1f}MB)")
                    
                    return {
                        'max_cand': max_cand,
                        'boxes': total_boxes,
                        'csv': str(dest),
                        'success': True
                    }
                    
    except Exception as e:
        print(f"✗ 에러: {e}")
    
    return {'max_cand': max_cand, 'success': False}

def main():
    print("="*80)
    print("Max Candidates 조정 실험")
    print("="*80)
    print("목표: 더 많은 후보 검출로 Recall 향상")
    print(f"조합: {len(PARAM_GRID)}개 (약 {len(PARAM_GRID)}분 소요)")
    print()
    
    results = []
    for idx, (thresh, box_thresh, max_cand) in enumerate(PARAM_GRID):
        result = run_single_prediction(thresh, box_thresh, max_cand, idx, len(PARAM_GRID))
        results.append(result)
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    if successful:
        print("\n생성된 파일:")
        for r in successful:
            print(f"  max_cand={r['max_cand']:>4}: {r['boxes']:>6,} boxes → {Path(r['csv']).name}")

if __name__ == "__main__":
    main()
