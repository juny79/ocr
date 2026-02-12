#!/usr/bin/env python3
"""
Fold 3 후처리 파라미터 그리드 서치
빠른 실행을 위한 간소화 버전
"""
import os
import sys
import json
import lightning.pytorch as pl
from pathlib import Path
from datetime import datetime
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

# 그리드 파라미터 (3개만 빠르게 테스트)
PARAM_GRID = [
    (0.210, 0.390),  # 낮춤
    (0.215, 0.395),  # 약간 낮춤
    (0.225, 0.405),  # 약간 높임
    (0.230, 0.410),  # 높임  
    (0.218, 0.398),  # 세밀 조정 1
    (0.222, 0.402),  # 세밀 조정 2
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_grid_search")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_single_prediction(thresh, box_thresh, idx):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{len(PARAM_GRID)}] thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
    print('='*80)
    
    # Hydra 초기화
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
            
            # 출력 디렉토리
            output_name = f"fold3_t{int(thresh*1000)}_b{int(box_thresh*1000)}"
            cfg.submission_dir = str(OUTPUT_BASE / output_name)
            
            # 모델 및 데이터 생성
            from ocr.lightning_modules import get_pl_modules_by_cfg
            model_module, data_module = get_pl_modules_by_cfg(cfg)
            
            # Trainer 생성 및 예측
            trainer = pl.Trainer(logger=False, devices=1)
            trainer.predict(model_module, data_module, ckpt_path=CHECKPOINT)
            
            # JSON 파일 경로
            json_file = Path(cfg.submission_dir) / "predictions.json"
            
            if json_file.exists():
                # 박스 수 확인
                with open(json_file, 'r') as f:
                    data = json.load(f)
                total_boxes = sum(len(img_data.get('words', {})) for img_data in data['images'].values())
                print(f"✓ 예측 완료: {total_boxes:,} boxes")
                
                # CSV 변환
                csv_file = OUTPUT_BASE / f"{output_name}.csv"
                from ocr.utils.convert_submission import convert_json_to_csv
                result = convert_json_to_csv(str(json_file), str(csv_file))
                
                if result:
                    print(f"✓ CSV 저장: {csv_file}")
                    return {
                        'thresh': thresh,
                        'box_thresh': box_thresh,
                        'boxes': total_boxes,
                        'json': str(json_file),
                        'csv': str(csv_file),
                        'success': True
                    }
            else:
                print(f"✗ 예측 파일 없음")
                
    except Exception as e:
        print(f"✗ 에러: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'thresh': thresh,
        'box_thresh': box_thresh,
        'success': False
    }

def main():
    """메인 실행"""
    print("="*80)
    print("Fold 3 후처리 파라미터 그리드 서치")
    print("="*80)
    print(f"체크포인트: {CHECKPOINT}")
    print(f"총 {len(PARAM_GRID)}개 조합 테스트")
    print(f"기준점: thresh=0.220, box_thresh=0.400 → H-Mean=0.9863")
    print()
    
    results = []
    
    for idx, (thresh, box_thresh) in enumerate(PARAM_GRID):
        result = run_single_prediction(thresh, box_thresh, idx)
        results.append(result)
        
        # 중간 저장
        results_file = OUTPUT_BASE / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("그리드 서치 완료!")
    print("="*80)
    print(f"성공: {sum(1 for r in results if r.get('success'))} / {len(PARAM_GRID)}")
    print()
    print("생성된 CSV 파일:")
    for r in results:
        if r.get('csv'):
            print(f"  {r['thresh']:.3f}, {r['box_thresh']:.3f} → {r['csv']}")
    print()
    print("다음 단계:")
    print("1. 각 CSV 파일을 리더보드에 제출")
    print("2. 0.9863을 초과하는 파라미터 확인")
    print("3. 최고 점수 파일 사용")

if __name__ == "__main__":
    main()
