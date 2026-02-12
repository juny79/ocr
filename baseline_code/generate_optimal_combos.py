#!/usr/bin/env python3
"""
최적 조합 생성: 낮은 thresh + unclip_ratio 조정
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

# 최적 조합: 낮은 thresh (Recall 상승) + 적정 unclip (Precision 균형)
OPTIMAL_COMBOS = [
    # thresh를 낮춰서 Recall 상승 + unclip_ratio로 Precision 조절
    (0.212, 0.392, 1.9, "최우선 - Recall 최대"),
    (0.212, 0.392, 2.0, "최우선 - 기준"),
    (0.215, 0.395, 1.9, "2순위 - 안전"),
    (0.215, 0.395, 2.0, "2순위 - 기준"),
    (0.218, 0.398, 1.9, "3순위 - 검증"),
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_optimal")
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
            
            output_name = f"fold3_opt_t{int(thresh*1000)}_b{int(box_thresh*1000)}_u{int(unclip_ratio*10)}"
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
                        'box_thresh': box_thresh,
                        'unclip_ratio': unclip_ratio,
                        'boxes': total_boxes,
                        'csv': str(dest),
                        'success': True
                    }
                    
    except Exception as e:
        print(f"✗ 에러: {e}")
    
    return {'success': False}

def main():
    print("="*80)
    print("최적 조합 생성: 낮은 thresh + unclip_ratio 조정")
    print("="*80)
    print(f"총 조합: {len(OPTIMAL_COMBOS)}개")
    print(f"예상 소요 시간: {len(OPTIMAL_COMBOS)} 분")
    print()
    print("전략:")
    print("  1) thresh를 낮춰서 Recall 상승 (0.9828 → 0.9840+)")
    print("  2) unclip_ratio로 Precision 균형 (0.9860-0.9870)")
    print("  3) 목표: H-Mean 0.9863-0.9867 달성")
    print()
    
    results = []
    for idx, (thresh, box_thresh, unclip, desc) in enumerate(OPTIMAL_COMBOS):
        result = run_prediction(thresh, box_thresh, unclip, desc, idx, len(OPTIMAL_COMBOS))
        results.append(result)
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    if successful:
        print(f"\n성공: {len(successful)}/{len(OPTIMAL_COMBOS)}개")
        print("\n생성된 파일:")
        for r in successful:
            print(f"  {Path(r['csv']).name:<45} {r['boxes']:>6,} boxes")
        
        print("\n제출 우선순위:")
        print("  1순위: fold3_opt_t212_b392_u19.csv ⭐⭐⭐⭐⭐")
        print("  2순위: fold3_opt_t212_b392_u20.csv ⭐⭐⭐⭐")
        print("  3순위: fold3_opt_t215_b395_u19.csv ⭐⭐⭐")

if __name__ == "__main__":
    main()
