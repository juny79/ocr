#!/usr/bin/env python3
"""
Fold 3 후처리 파라미터 그리드 서치
체크포인트: checkpoints/kfold_optimized/fold_3/fold3_best.ckpt
목표: 0.9863 초과
"""
import os
import sys
import json
import lightning.pytorch as pl
from pathlib import Path
from datetime import datetime
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

# 그리드 파라미터 설정 (0.220, 0.400 주변 탐색)
THRESH_VALUES = [0.200, 0.210, 0.215, 0.220, 0.225, 0.230, 0.235, 0.240]
BOX_THRESH_VALUES = [0.380, 0.390, 0.395, 0.400, 0.405, 0.410, 0.420]

# 체크포인트 경로
CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"

# 출력 디렉토리
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_grid_search")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

# 결과 저장
RESULTS_FILE = OUTPUT_BASE / f"grid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def run_prediction(thresh, box_thresh, output_name):
    """
    특정 파라미터로 예측 수행
    
    Args:
        thresh: text threshold
        box_thresh: box threshold
        output_name: 출력 파일명
        
    Returns:
        bool: 성공 여부
    """
    output_dir = OUTPUT_BASE / output_name
    
    cmd = [
        "python", "runners/test.py",
        f"--config-name={CONFIG}",
        f"checkpoint_path={CHECKPOINT}",
        f"models.head.postprocess.thresh={thresh}",
        f"models.head.postprocess.box_thresh={box_thresh}",
        f"output_dir={output_dir}",
        f"fold={FOLD_NUM}",
    ]
    
    print(f"\n{'='*80}")
    print(f"실행: thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
    print(f"출력: {output_dir}")
    print('='*80)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10분 타임아웃
            cwd="/data/ephemeral/home/baseline_code"
        )
        
        if result.returncode == 0:
            print("✓ 예측 성공")
            return True
        else:
            print(f"✗ 예측 실패 (code {result.returncode})")
            print(result.stderr[-500:] if result.stderr else "")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 타임아웃")
        return False
    except Exception as e:
        print(f"✗ 에러: {e}")
        return False


def convert_to_csv(json_path, csv_path):
    """JSON을 CSV로 변환"""
    try:
        cmd = [
            "python", "-m", "ocr.utils.convert_submission",
            "-J", str(json_path),
            "-O", str(csv_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input="yes\n",  # 덮어쓰기 확인
            timeout=60,
            cwd="/data/ephemeral/home/baseline_code"
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"CSV 변환 실패: {result.stderr[-200:]}")
            return False
            
    except Exception as e:
        print(f"CSV 변환 에러: {e}")
        return False


def main():
    """메인 그리드 서치"""
    print("="*80)
    print("Fold 3 후처리 파라미터 그리드 서치")
    print("="*80)
    print(f"체크포인트: {CHECKPOINT}")
    print(f"Thresh 범위: {THRESH_VALUES}")
    print(f"Box Thresh 범위: {BOX_THRESH_VALUES}")
    print(f"총 조합: {len(THRESH_VALUES)} × {len(BOX_THRESH_VALUES)} = {len(THRESH_VALUES) * len(BOX_THRESH_VALUES)}")
    print(f"결과 저장: {RESULTS_FILE}")
    print()
    
    # 결과 저장용
    results = []
    
    # 기준점 (현재 최고 점수)
    baseline = {
        'thresh': 0.220,
        'box_thresh': 0.400,
        'hmean': 0.9863,
        'note': 'baseline (kfold_fold3_optimized_t0.220_b0.400_79.csv)'
    }
    results.append(baseline)
    
    print(f"기준점: thresh={baseline['thresh']}, box_thresh={baseline['box_thresh']}, H-Mean={baseline['hmean']}")
    print()
    
    # 그리드 서치
    total = len(THRESH_VALUES) * len(BOX_THRESH_VALUES)
    count = 0
    
    for thresh in THRESH_VALUES:
        for box_thresh in BOX_THRESH_VALUES:
            count += 1
            print(f"\n[{count}/{total}] thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
            
            output_name = f"fold3_t{thresh:.3f}_b{box_thresh:.3f}".replace('.', '')
            
            # 예측 수행
            success = run_prediction(thresh, box_thresh, output_name)
            
            if success:
                # JSON 파일 경로
                json_file = OUTPUT_BASE / output_name / "predictions.json"
                
                if json_file.exists():
                    # CSV 변환
                    csv_file = OUTPUT_BASE / f"{output_name}.csv"
                    csv_success = convert_to_csv(json_file, csv_file)
                    
                    result_entry = {
                        'thresh': thresh,
                        'box_thresh': box_thresh,
                        'json_path': str(json_file),
                        'csv_path': str(csv_file) if csv_success else None,
                        'success': True
                    }
                    results.append(result_entry)
                    
                    # 중간 결과 저장
                    with open(RESULTS_FILE, 'w') as f:
                        json.dump(results, f, indent=2)
                else:
                    print(f"✗ 예측 파일 없음: {json_file}")
            else:
                result_entry = {
                    'thresh': thresh,
                    'box_thresh': box_thresh,
                    'success': False,
                    'error': 'prediction failed'
                }
                results.append(result_entry)
    
    # 최종 결과 저장
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("그리드 서치 완료!")
    print("="*80)
    print(f"결과 저장: {RESULTS_FILE}")
    print(f"성공: {sum(1 for r in results if r.get('success', False))} / {total}")
    print()
    print("생성된 CSV 파일들:")
    for r in results:
        if r.get('csv_path'):
            print(f"  - {r['csv_path']}")
    print()
    print("제출 방법:")
    print("1. 각 CSV 파일을 리더보드에 제출")
    print("2. 가장 높은 점수의 파라미터 확인")
    print("3. 0.9863 초과 시 성공!")
    

if __name__ == "__main__":
    main()
