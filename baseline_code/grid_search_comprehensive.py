#!/usr/bin/env python3
"""
확장 그리드 서치: thresh 0.200~0.240, box_thresh 0.380~0.420
다양한 조합으로 최적점 탐색
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

# 확장된 그리드: 더 넓은 범위, 더 촘촘한 간격
# thresh를 0.003 간격으로 0.200~0.240 전체 커버
PARAM_GRID = [
    # 낮은 구간 (0.200-0.212) - 6개
    (0.200, 0.380),
    (0.203, 0.383),
    (0.206, 0.386),
    (0.209, 0.389),
    (0.212, 0.392),
    (0.215, 0.395),
    
    # 중간 구간 (0.218-0.224) - 3개 (현재 peak 근처)
    (0.218, 0.398),
    (0.221, 0.401),
    (0.224, 0.404),
    
    # 중고 구간 (0.227-0.233) - 3개
    (0.227, 0.407),
    (0.230, 0.410),
    (0.233, 0.413),
    
    # 높은 구간 (0.236-0.240) - 2개
    (0.236, 0.416),
    (0.240, 0.420),
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_comprehensive")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_single_prediction(thresh, box_thresh, idx, total):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
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
                
                csv_file = OUTPUT_BASE / f"{output_name}_wide.csv"
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
                        'boxes': total_boxes, 
                        'size_mb': file_size_mb,
                        'csv': str(dest), 
                        'success': True
                    }
                    
    except Exception as e:
        print(f"✗ 에러: {e}")
    
    return {'thresh': thresh, 'box_thresh': box_thresh, 'success': False}

def main():
    print("="*80)
    print("확장 그리드 서치: 0.200~0.240 전체 범위 탐색")
    print("="*80)
    print(f"총 조합: {len(PARAM_GRID)}개")
    print(f"예상 소요 시간: {len(PARAM_GRID)} 분")
    print()
    print("현재까지 발견된 패턴:")
    print("  0.210/0.390 → H-Mean 0.9858")
    print("  0.218/0.398 → H-Mean 0.9860 (현재 최고)")
    print("  0.225/0.405 → H-Mean 0.9855")
    print()
    print("탐색 전략:")
    print("  1) 낮은 범위 (0.200-0.215): peak 왼쪽 영역")
    print("  2) Peak 근처 (0.218-0.224): 세밀 탐색")
    print("  3) 높은 범위 (0.227-0.240): 우측 경계 확인")
    print()
    
    results = []
    start_time = datetime.now()
    
    for idx, (thresh, box_thresh) in enumerate(PARAM_GRID):
        result = run_single_prediction(thresh, box_thresh, idx, len(PARAM_GRID))
        results.append(result)
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        remaining = (len(PARAM_GRID) - idx - 1) * (elapsed / (idx + 1))
        print(f"경과: {elapsed:.1f}분, 예상 남은 시간: {remaining:.1f}분")
    
    print("\n" + "="*80)
    print("그리드 서치 완료!")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    print(f"\n성공: {len(successful)}/{len(PARAM_GRID)}개")
    print("\n생성된 파일 목록:")
    
    for r in successful:
        csv_name = Path(r['csv']).name
        print(f"  {csv_name:<45} {r['boxes']:>6,} boxes ({r['size_mb']:.1f}MB)")
    
    # 박스 수 기준 분석
    if successful:
        print("\n박스 수 분포:")
        boxes_sorted = sorted(successful, key=lambda x: x['boxes'])
        print(f"  최소: {boxes_sorted[0]['boxes']:,} (thresh={boxes_sorted[0]['thresh']:.3f})")
        print(f"  최대: {boxes_sorted[-1]['boxes']:,} (thresh={boxes_sorted[-1]['thresh']:.3f})")
        print(f"  평균: {sum(r['boxes'] for r in successful)//len(successful):,}")
    
    print(f"\n전체 소요 시간: {(datetime.now() - start_time).total_seconds() / 60:.1f}분")
    print("\n다음 단계:")
    print("  1) 각 파일을 제출하여 H-Mean 측정")
    print("  2) 결과를 바탕으로 최적 조합 선택")
    print("  3) 최적점 주변 ±0.001 범위 추가 탐색 고려")

if __name__ == "__main__":
    main()
