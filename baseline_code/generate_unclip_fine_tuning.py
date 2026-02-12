#!/usr/bin/env python3
"""
미세 조정: unclip_ratio 정밀 탐색
- 0.220 기준: unclip 1.95, 1.98
- 0.218 기준: unclip 1.97, 2.0
"""
import os
import sys
import json
import shutil
import lightning.pytorch as pl
from pathlib import Path
from datetime import datetime
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

# 4개 조합
PARAM_GRID = [
    # 0.220 기준 (unclip=1.9가 최적이었음, 주변 탐색)
    (0.220, 0.400, 1.95, "기준점 A (1.9와 2.0 중간)"),
    (0.220, 0.400, 1.98, "기준점 B (2.0 근처)"),
    
    # 0.218 기준 (현재 최고점, unclip 미세 조정)
    (0.218, 0.398, 1.97, "Peak A (2.0 근처)"),
    (0.218, 0.398, 2.00, "Peak B (기본)"),
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_unclip_fine")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_single_prediction(thresh, box_thresh, unclip_ratio, description, idx, total):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] {description}")
    print(f"  thresh={thresh:.3f}, box_thresh={box_thresh:.3f}, unclip={unclip_ratio:.2f}")
    print(f"진행률: {(idx+1)/total*100:.1f}%")
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
            
            output_name = f"fold3_t{int(thresh*1000)}_b{int(box_thresh*1000)}_u{int(unclip_ratio*100)}"
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
                    print(f"✓ 제출 폴더 복사 완료")
                    
                    return {
                        'thresh': thresh,
                        'box_thresh': box_thresh,
                        'unclip_ratio': unclip_ratio,
                        'boxes': total_boxes,
                        'size_mb': file_size_mb,
                        'csv': str(dest),
                        'success': True
                    }
                    
    except Exception as e:
        print(f"✗ 에러: {e}")
        import traceback
        traceback.print_exc()
    
    return {'unclip_ratio': unclip_ratio, 'success': False}

def main():
    print("="*80)
    print("Unclip Ratio 미세 조정")
    print("="*80)
    print(f"총 조합: {len(PARAM_GRID)}개")
    print(f"예상 소요 시간: {len(PARAM_GRID)} 분")
    print()
    print("전략:")
    print("  그룹 1) thresh=0.220 (unclip=1.9가 최적)")
    print("    - unclip=1.95 (1.9와 2.0 중간)")
    print("    - unclip=1.98 (2.0 근처)")
    print()
    print("  그룹 2) thresh=0.218 (현재 최고 H=0.9860)")
    print("    - unclip=1.97 (미세 조정)")
    print("    - unclip=2.00 (기본, 재확인)")
    print()
    print("목표: H-Mean 0.9863+ 달성")
    print()
    
    results = []
    start_time = datetime.now()
    
    for idx, (thresh, box_thresh, unclip_ratio, desc) in enumerate(PARAM_GRID):
        result = run_single_prediction(thresh, box_thresh, unclip_ratio, desc, idx, len(PARAM_GRID))
        results.append(result)
        
        if idx > 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            remaining = (len(PARAM_GRID) - idx - 1) * (elapsed / (idx + 1))
            print(f"경과: {elapsed:.1f}분, 예상 남은 시간: {remaining:.1f}분")
    
    print("\n" + "="*80)
    print("Unclip Ratio 미세 조정 완료!")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    print(f"\n성공: {len(successful)}/{len(PARAM_GRID)}개")
    
    if successful:
        print("\n생성된 파일:")
        for r in successful:
            csv_name = Path(r['csv']).name
            print(f"  {csv_name:<50} {r['boxes']:>6,} boxes ({r['size_mb']:.1f}MB)")
        
        print("\n파라미터 요약:")
        print("\nthresh=0.220 기준:")
        for r in successful:
            if r['thresh'] == 0.220:
                print(f"  unclip={r['unclip_ratio']:.2f}: {r['boxes']:,} boxes")
        
        print("\nthresh=0.218 기준:")
        for r in successful:
            if r['thresh'] == 0.218:
                print(f"  unclip={r['unclip_ratio']:.2f}: {r['boxes']:,} boxes")
    
    print(f"\n전체 소요 시간: {(datetime.now() - start_time).total_seconds() / 60:.1f}분")
    print("\n✅ 다음 단계:")
    print("  1) 각 파일을 리더보드에 제출")
    print("  2) unclip_ratio 미세 조정 효과 분석")
    print("  3) 최적 조합 발견!")
    print()
    print("제출 우선순위:")
    print("  1순위: fold3_t220_b400_u195.csv (0.220 기준 중간값)")
    print("  2순위: fold3_t218_b398_u197.csv (0.218 기준 미세 조정)")
    print("  3순위: fold3_t220_b400_u198.csv")
    print("  4순위: fold3_t218_b398_u200.csv (재확인)")

if __name__ == "__main__":
    main()
