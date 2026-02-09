"""
간소화된 후처리 파라미터 그리드 서치
Hydra config를 사용하여 모델 로드
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg

# 현재 최고 점수의 파라미터
BASELINE_THRESH = 0.231
BASELINE_BOX_THRESH = 0.432

# 그리드 서치 범위 설정
THRESH_RANGE = np.arange(0.22, 0.25, 0.005)  # 0.22 ~ 0.245, step 0.005 (좁은 범위)
BOX_THRESH_RANGE = np.arange(0.41, 0.45, 0.005)  # 0.41 ~ 0.445, step 0.005

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='test', version_base='1.2')
def main(config: DictConfig):
    """그리드 서치 메인 함수"""
    print("="*80)
    print("후처리 파라미터 그리드 서치")
    print("="*80)
    print(f"Thresh range: {THRESH_RANGE[0]:.3f} ~ {THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Box Thresh range: {BOX_THRESH_RANGE[0]:.3f} ~ {BOX_THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Total combinations: {len(THRESH_RANGE)} × {len(BOX_THRESH_RANGE)} = {len(THRESH_RANGE) * len(BOX_THRESH_RANGE)}")
    print(f"Baseline params: thresh={BASELINE_THRESH:.3f}, box_thresh={BASELINE_BOX_THRESH:.3f}")
    print()
    
    # 결과 저장 디렉토리
    results_dir = Path("/data/ephemeral/home/baseline_code/grid_search_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Checkpoint 설정
    checkpoint_path = "outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt"
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
    config.ckpt_path = checkpoint_path
    OmegaConf.set_struct(config, True)
    
    # 결과 저장용
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
        'experiments': []
    }
    
    best_hmean = 0.0
    best_params = None
    
    # 그리드 서치
    total = len(THRESH_RANGE) * len(BOX_THRESH_RANGE)
    
    with tqdm(total=total, desc="Grid Search Progress") as pbar:
        for thresh in THRESH_RANGE:
            for box_thresh in BOX_THRESH_RANGE:
                try:
                    # 후처리 파라미터 설정
                    from omegaconf import OmegaConf
                    OmegaConf.set_struct(config, False)
                    config.models.head.postprocess.thresh = float(thresh)
                    config.models.head.postprocess.box_thresh = float(box_thresh)
                    OmegaConf.set_struct(config, True)
                    
                    # 모델 및 데이터 로더 생성
                    pl.seed_everything(config.get("seed", 42), workers=True)
                    model_module, data_module = get_pl_modules_by_cfg(config)
                    
                    # Trainer 생성
                    trainer = pl.Trainer(
                        logger=False,
                        enable_checkpointing=False,
                        enable_progress_bar=False,
                        enable_model_summary=False
                    )
                    
                    # 테스트 실행
                    test_results = trainer.test(
                        model_module,
                        data_module,
                        ckpt_path=checkpoint_path,
                        verbose=False
                    )
                    
                    # 결과 추출
                    if test_results and len(test_results) > 0:
                        metrics = test_results[0]
                        hmean = metrics.get('test/hmean', 0.0)
                        precision = metrics.get('test/precision', 0.0)
                        recall = metrics.get('test/recall', 0.0)
                        
                        experiment_result = {
                            'thresh': float(thresh),
                            'box_thresh': float(box_thresh),
                            'metrics': {
                                'hmean': float(hmean),
                                'precision': float(precision),
                                'recall': float(recall),
                                'success': True
                            }
                        }
                        
                        results['experiments'].append(experiment_result)
                        
                        # 최고 점수 업데이트
                        if hmean > best_hmean:
                            best_hmean = hmean
                            best_params = {
                                'thresh': float(thresh),
                                'box_thresh': float(box_thresh),
                                'hmean': float(hmean),
                                'precision': float(precision),
                                'recall': float(recall)
                            }
                            pbar.set_postfix({'best_hmean': f"{best_hmean:.6f}"})
                    
                except Exception as e:
                    print(f"\nError at thresh={thresh:.3f}, box_thresh={box_thresh:.3f}: {e}")
                    experiment_result = {
                        'thresh': float(thresh),
                        'box_thresh': float(box_thresh),
                        'metrics': {
                            'hmean': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'success': False,
                            'error': str(e)
                        }
                    }
                    results['experiments'].append(experiment_result)
                
                pbar.update(1)
                
                # 중간 저장
                if len(results['experiments']) % 20 == 0:
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
    
    # 최종 결과 저장
    results['best'] = best_params
    results['completed_at'] = datetime.now().isoformat()
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 결과 출력
    print("\n" + "="*80)
    print("그리드 서치 완료!")
    print("="*80)
    print(f"결과 파일: {results_file}")
    
    if best_params:
        print(f"\n🏆 최고 성능 파라미터:")
        print(f"  thresh: {best_params['thresh']:.4f}")
        print(f"  box_thresh: {best_params['box_thresh']:.4f}")
        print(f"  H-Mean: {best_params['hmean']:.6f}")
        print(f"  Precision: {best_params['precision']:.6f}")
        print(f"  Recall: {best_params['recall']:.6f}")
    
    # 상위 5개
    sorted_results = sorted(
        [r for r in results['experiments'] if r['metrics']['success']],
        key=lambda x: x['metrics']['hmean'],
        reverse=True
    )[:5]
    
    print(f"\n📈 상위 5개 결과:")
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. thresh={result['thresh']:.4f}, box_thresh={result['box_thresh']:.4f} "
              f"→ H-Mean: {result['metrics']['hmean']:.6f}")


if __name__ == "__main__":
    main()
