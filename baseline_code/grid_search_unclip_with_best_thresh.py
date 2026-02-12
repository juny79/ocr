#!/usr/bin/env python3
"""
Unclip Ratio 그리드 서치 (코드 패치 적용 버전)
thresh=0.218, box_thresh=0.398 고정 + unclip_ratio 탐색
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

# Unclip Ratio 탐색 (0.9863 기준점 파라미터 사용!)
PARAM_GRID = [
    (0.220, 0.400, 1.8),   # 보수적
    (0.220, 0.400, 1.9),   
    (0.220, 0.400, 2.0),   # 현재 기본값 (0.9863)
    (0.220, 0.400, 2.1),   
    (0.220, 0.400, 2.2),   # 적극적
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_unclip_grid")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_single_prediction(thresh, box_thresh, unclip_ratio, idx, total):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] unclip_ratio={unclip_ratio:.1f}")
    print(f"  thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
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
                f'models.head.postprocess.unclip_ratio={unclip_ratio}',  # ✅ 패치 후 가능!
            ])
            cfg.checkpoint_path = CHECKPOINT
            cfg.minified_json = False
            
            output_name = f"fold3_unclip{int(unclip_ratio*10)}_t{int(thresh*1000)}_b{int(box_thresh*1000)}"
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
                        'unclip_ratio': unclip_ratio,
                        'thresh': thresh,
                        'box_thresh': box_thresh,
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
    print("Unclip Ratio 그리드 서치")
    print("="*80)
    print(f"총 조합: {len(PARAM_GRID)}개")
    print(f"예상 소요 시간: {len(PARAM_GRID)} 분")
    print()
    print("전략:")
    print("  - 기준점 파라미터 고정: thresh=0.220, box_thresh=0.400 (0.9863)")
    print("  - unclip_ratio만 변경 (1.8 ~ 2.2)")
    print("  - 목표: Recall 향상으로 0.9863+ 달성")
    print()
    print("기준점 (unclip_ratio=2.0):")
    print("  H-Mean 0.9863")
    print()
    print("현재 문제 (thresh 높일 경우):")
    print("  H-Mean 0.9843 (P=0.9888, R=0.9806)")
    print("  → Recall이 40포인트 부족!")
    print()
    
    results = []
    start_time = datetime.now()
    
    for idx, (thresh, box_thresh, unclip_ratio) in enumerate(PARAM_GRID):
        result = run_single_prediction(thresh, box_thresh, unclip_ratio, idx, len(PARAM_GRID))
        results.append(result)
        
        if idx > 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            remaining = (len(PARAM_GRID) - idx - 1) * (elapsed / (idx + 1))
            print(f"경과: {elapsed:.1f}분, 예상 남은 시간: {remaining:.1f}분")
    
    print("\n" + "="*80)
    print("Unclip Ratio 그리드 서치 완료!")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    print(f"\n성공: {len(successful)}/{len(PARAM_GRID)}개")
    
    if successful:
        print("\n생성된 파일:")
        for r in successful:
            csv_name = Path(r['csv']).name
            print(f"  {csv_name:<50} {r['boxes']:>6,} boxes")
        
        print("\nUnclip Ratio 별 박스 수:")
        for r in successful:
            print(f"  unclip_ratio={r['unclip_ratio']:.1f}: {r['boxes']:,} boxes")
    
    print(f"\n전체 소요 시간: {(datetime.now() - start_time).total_seconds() / 60:.1f}분")
    print("\n✅ 다음 단계:")
    print("  1) 각 파일을 리더보드에 제출")
    print("  2) unclip_ratio와 H-Mean 상관관계 분석")
    print("  3) 최적 unclip_ratio 발견!")

if __name__ == "__main__":
    main()
