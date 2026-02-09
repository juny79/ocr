"""
빠른 후처리 파라미터 그리드 서치
모델을 한 번만 로드하고 후처리 파라미터만 변경하여 빠르게 탐색
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import lightning.pytorch as pl
from omegaconf import OmegaConf

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.metrics.cleval_metric import CLEvalMetric

# 현재 최고 점수의 파라미터
BASELINE_THRESH = 0.231
BASELINE_BOX_THRESH = 0.432

# 그리드 서치 범위 설정 (더 세밀하게)
THRESH_RANGE = np.arange(0.21, 0.26, 0.005)  # 0.21 ~ 0.255, step 0.005
BOX_THRESH_RANGE = np.arange(0.40, 0.46, 0.005)  # 0.40 ~ 0.455, step 0.005

# 모델 체크포인트 경로
CHECKPOINT_PATH = "outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt"

# 결과 저장 디렉토리
RESULTS_DIR = Path("grid_search_results")
RESULTS_DIR.mkdir(exist_ok=True)

# 결과 저장 파일
RESULTS_FILE = RESULTS_DIR / f"grid_search_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def load_model_and_data():
    """모델과 데이터 로더 준비"""
    print("모델 로딩 중...")
    
    # 설정 로드
    config_path = Path("configs/preset/hrnet_w44_1024.yaml")
    config = OmegaConf.load(config_path)
    
    # Base config 병합
    base_config = OmegaConf.load("configs/preset/base.yaml")
    config = OmegaConf.merge(base_config, config)
    
    # 필요한 설정 추가
    config.seed = 42
    config.checkpoint_path = CHECKPOINT_PATH
    
    pl.seed_everything(42, workers=True)
    
    # 모델과 데이터 로더 생성
    model_module, data_module = get_pl_modules_by_cfg(config)
    
    # 체크포인트 로드
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model_module.load_state_dict(checkpoint['state_dict'])
    model_module.eval()
    model_module.cuda()
    
    # 데이터 로더 준비
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    
    return model_module, test_dataloader, config


def evaluate_with_params(model, dataloader, thresh, box_thresh):
    """
    특정 후처리 파라미터로 평가
    
    Args:
        model: 학습된 모델
        dataloader: 테스트 데이터로더
        thresh: threshold 값
        box_thresh: box threshold 값
        
    Returns:
        dict: 평가 결과
    """
    # 후처리 파라미터 업데이트
    model.model.head.postprocessor.thresh = thresh
    model.model.head.postprocessor.box_thresh = box_thresh
    
    # 메트릭 초기화
    metric = CLEvalMetric()
    
    # 예측 수행
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # GPU로 이동
            images = batch['images'].cuda()
            
            # 예측
            outputs = model(images)
            
            # 메트릭 업데이트
            metric.update(
                outputs,
                batch['gt_polygons'],
                batch['gt_ignore_masks']
            )
    
    # 최종 메트릭 계산
    results = metric.compute()
    
    return {
        'hmean': float(results['hmean']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'success': True
    }


def main():
    """그리드 서치 메인 함수"""
    print("="*80)
    print("빠른 후처리 파라미터 그리드 서치 시작")
    print("="*80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Thresh range: {THRESH_RANGE[0]:.3f} ~ {THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Box Thresh range: {BOX_THRESH_RANGE[0]:.3f} ~ {BOX_THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Total combinations: {len(THRESH_RANGE)} × {len(BOX_THRESH_RANGE)} = {len(THRESH_RANGE) * len(BOX_THRESH_RANGE)}")
    print(f"Baseline params: thresh={BASELINE_THRESH:.3f}, box_thresh={BASELINE_BOX_THRESH:.3f}")
    print()
    
    # 모델과 데이터 로드
    try:
        model, dataloader, config = load_model_and_data()
        print("✓ 모델 및 데이터 로드 완료\n")
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        return
    
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
    
    # 그리드 서치 수행
    with tqdm(total=total_experiments, desc="Grid Search") as pbar:
        for thresh in THRESH_RANGE:
            for box_thresh in BOX_THRESH_RANGE:
                try:
                    # 평가 실행
                    metrics = evaluate_with_params(model, dataloader, thresh, box_thresh)
                    
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
                        pbar.set_postfix({
                            'best_hmean': f"{best_hmean:.6f}",
                            'thresh': f"{thresh:.3f}",
                            'box_thresh': f"{box_thresh:.3f}"
                        })
                
                except Exception as e:
                    print(f"\n✗ Error at thresh={thresh:.3f}, box_thresh={box_thresh:.3f}: {e}")
                    experiment_result = {
                        'thresh': float(thresh),
                        'box_thresh': float(box_thresh),
                        'metrics': {
                            'hmean': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'success': False,
                            'error': str(e)
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    results['experiments'].append(experiment_result)
                
                pbar.update(1)
                
                # 중간 결과 저장 (50개마다)
                if len(results['experiments']) % 50 == 0:
                    results['best_so_far'] = best_params
                    with open(RESULTS_FILE, 'w') as f:
                        json.dump(results, f, indent=2)
    
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
        baseline_val_hmean = best_params['hmean']  # validation
        baseline_sub_hmean = 0.9851  # submission
        improvement = best_params['hmean'] - baseline_val_hmean
        print(f"\n📊 Baseline submission 대비 (참고용):")
        print(f"  Submission H-Mean: {baseline_sub_hmean:.4f}")
        print(f"  Validation H-Mean (이번 최고): {best_params['hmean']:.6f}")
    
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
    
    # 히트맵 데이터 생성 (시각화용)
    print("\n생성된 히트맵 데이터를 사용하여 시각화할 수 있습니다.")
    heatmap_data = np.zeros((len(THRESH_RANGE), len(BOX_THRESH_RANGE)))
    for i, thresh in enumerate(THRESH_RANGE):
        for j, box_thresh in enumerate(BOX_THRESH_RANGE):
            for exp in results['experiments']:
                if (abs(exp['thresh'] - thresh) < 0.001 and 
                    abs(exp['box_thresh'] - box_thresh) < 0.001 and
                    exp['metrics']['success']):
                    heatmap_data[i, j] = exp['metrics']['hmean']
                    break
    
    # 히트맵 저장
    heatmap_file = RESULTS_DIR / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    np.save(heatmap_file, heatmap_data)
    print(f"히트맵 데이터 저장: {heatmap_file}")


if __name__ == "__main__":
    main()
