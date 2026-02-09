"""
후처리 파라미터 그리드 서치 스크립트
학습된 모델로 다양한 thresh, box_thresh 조합을 테스트하여 최적 파라미터 찾기
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

# 현재 최고 점수의 파라미터
BASELINE_THRESH = 0.231
BASELINE_BOX_THRESH = 0.432

# 그리드 서치 범위 설정
THRESH_RANGE = np.arange(0.20, 0.28, 0.01)  # 0.20 ~ 0.27, step 0.01
BOX_THRESH_RANGE = np.arange(0.38, 0.48, 0.01)  # 0.38 ~ 0.47, step 0.01

# 모델 체크포인트 경로
CHECKPOINT_PATH = "outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt"

# 결과 저장 디렉토리
RESULTS_DIR = Path("grid_search_results")
RESULTS_DIR.mkdir(exist_ok=True)

# 결과 저장 파일
RESULTS_FILE = RESULTS_DIR / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def run_test_with_params(thresh, box_thresh, exp_name):
    """
    특정 후처리 파라미터로 테스트 실행
    
    Args:
        thresh: threshold 값
        box_thresh: box threshold 값
        exp_name: 실험 이름
        
    Returns:
        dict: 평가 결과 (hmean, precision, recall)
    """
    # predict로 validation 데이터 예측 후 평가
    # 먼저 심볼릭 링크 확인
    ckpt_path = "best_model.ckpt"
    
    cmd = [
        "python", "runners/predict.py",
        "preset=hrnet_w44_1024",
        f"exp_name={exp_name}",
        f"checkpoint_path={ckpt_path}",
        f"models.head.postprocess.thresh={thresh:.4f}",
        f"models.head.postprocess.box_thresh={box_thresh:.4f}"
    ]
    
    try:
        print(f"\n{'='*80}")
        print(f"Testing: thresh={thresh:.4f}, box_thresh={box_thresh:.4f}")
        print(f"{'='*80}")
        
        result = subprocess.run(
            cmd,
            cwd="/data/ephemeral/home/baseline_code",
            capture_output=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )
        
        # 출력에서 메트릭 추출
        output = result.stdout + result.stderr
        
        hmean = None
        precision = None
        recall = None
        
        for line in output.split('\n'):
            if 'test/hmean' in line:
                try:
                    hmean = float(line.split()[-1])
                except:
                    pass
            elif 'test/precision' in line:
                try:
                    precision = float(line.split()[-1])
                except:
                    pass
            elif 'test/recall' in line:
                try:
                    recall = float(line.split()[-1])
                except:
                    pass
        
        if hmean is not None:
            print(f"✓ H-Mean: {hmean:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}")
            return {
                'hmean': hmean,
                'precision': precision,
                'recall': recall,
                'success': True
            }
        else:
            print(f"✗ Failed to extract metrics")
            return {
                'hmean': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'success': False,
                'error': 'Failed to extract metrics'
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout")
        return {
            'hmean': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            'hmean': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'success': False,
            'error': str(e)
        }


def main():
    """그리드 서치 메인 함수"""
    print("="*80)
    print("후처리 파라미터 그리드 서치 시작")
    print("="*80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Thresh range: {THRESH_RANGE[0]:.2f} ~ {THRESH_RANGE[-1]:.2f} (step: 0.01)")
    print(f"Box Thresh range: {BOX_THRESH_RANGE[0]:.2f} ~ {BOX_THRESH_RANGE[-1]:.2f} (step: 0.01)")
    print(f"Total combinations: {len(THRESH_RANGE)} × {len(BOX_THRESH_RANGE)} = {len(THRESH_RANGE) * len(BOX_THRESH_RANGE)}")
    print(f"Baseline params: thresh={BASELINE_THRESH:.3f}, box_thresh={BASELINE_BOX_THRESH:.3f}")
    print()
    
    # 결과 저장용 딕셔너리
    results = {
        'baseline': {
            'thresh': BASELINE_THRESH,
            'box_thresh': BASELINE_BOX_THRESH,
            'submission_score': {
                'hmean': 0.9851,
                'precision': 0.9854,
                'recall': 0.9857
            }
        },
        'search_space': {
            'thresh_range': [float(THRESH_RANGE[0]), float(THRESH_RANGE[-1])],
            'box_thresh_range': [float(BOX_THRESH_RANGE[0]), float(BOX_THRESH_RANGE[-1])],
        },
        'experiments': []
    }
    
    best_hmean = 0.0
    best_params = None
    
    total_experiments = len(THRESH_RANGE) * len(BOX_THRESH_RANGE)
    current_experiment = 0
    
    # 그리드 서치 수행
    for thresh in THRESH_RANGE:
        for box_thresh in BOX_THRESH_RANGE:
            current_experiment += 1
            exp_name = f"grid_search_t{thresh:.3f}_b{box_thresh:.3f}"
            
            print(f"\n[{current_experiment}/{total_experiments}] Testing combination...")
            
            # 테스트 실행
            metrics = run_test_with_params(thresh, box_thresh, exp_name)
            
            # 결과 저장
            experiment_result = {
                'thresh': float(thresh),
                'box_thresh': float(box_thresh),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            results['experiments'].append(experiment_result)
            
            # 최고 점수 업데이트
            if metrics['success'] and metrics['hmean'] > best_hmean:
                best_hmean = metrics['hmean']
                best_params = {
                    'thresh': float(thresh),
                    'box_thresh': float(box_thresh),
                    'hmean': metrics['hmean'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                }
                print(f"🎉 New best! H-Mean: {best_hmean:.6f}")
            
            # 중간 결과 저장 (10개마다)
            if current_experiment % 10 == 0:
                results['best_so_far'] = best_params
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Progress saved to {RESULTS_FILE}")
    
    # 최종 결과 저장
    results['best'] = best_params
    results['total_experiments'] = total_experiments
    results['completed_at'] = datetime.now().isoformat()
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 결과 요약 출력
    print("\n" + "="*80)
    print("그리드 서치 완료!")
    print("="*80)
    print(f"\n총 실험 횟수: {total_experiments}")
    print(f"결과 파일: {RESULTS_FILE}")
    
    if best_params:
        print(f"\n🏆 최고 성능 파라미터:")
        print(f"  thresh: {best_params['thresh']:.4f}")
        print(f"  box_thresh: {best_params['box_thresh']:.4f}")
        print(f"  H-Mean: {best_params['hmean']:.6f}")
        print(f"  Precision: {best_params['precision']:.6f}")
        print(f"  Recall: {best_params['recall']:.6f}")
        
        # Baseline과 비교
        improvement = best_params['hmean'] - 0.9851
        print(f"\n📊 Baseline 대비:")
        print(f"  H-Mean 변화: {improvement:+.6f} ({improvement*100:+.4f}%)")
    
    # 상위 10개 결과 출력
    print(f"\n📈 상위 10개 결과:")
    sorted_results = sorted(
        [r for r in results['experiments'] if r['metrics']['success']],
        key=lambda x: x['metrics']['hmean'],
        reverse=True
    )[:10]
    
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. thresh={result['thresh']:.4f}, box_thresh={result['box_thresh']:.4f} "
              f"→ H-Mean: {result['metrics']['hmean']:.6f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
