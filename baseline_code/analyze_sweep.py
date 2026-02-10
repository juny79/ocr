#!/usr/bin/env python
"""
WandB Sweep 결과 분석 및 최적 파라미터 추출
"""

import wandb
import os
import pandas as pd
from typing import Dict, List, Tuple

def fetch_sweep_results(entity: str, project: str, sweep_id: str) -> pd.DataFrame:
    """Sweep 결과를 WandB에서 가져오기"""
    
    api = wandb.Api(overrides={"entity": entity, "project": project})
    sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")
    
    runs_data = []
    for run in sweep.runs:
        run_data = {
            'run_name': run.name,
            'run_id': run.id,
            'state': run.state,
            'val_h_mean': run.summary.get('val_h_mean', None),
            'val_precision': run.summary.get('val_precision', None),
            'val_recall': run.summary.get('val_recall', None),
        }
        
        # 학습 파라미터
        if run.config:
            run_data.update({
                'lr': run.config.get('lr'),
                'weight_decay': run.config.get('weight_decay'),
                'T_max': run.config.get('T_max'),
                'thresh': run.config.get('thresh'),
                'box_thresh': run.config.get('box_thresh'),
                'batch_size': run.config.get('batch_size'),
                'max_epochs': run.config.get('max_epochs'),
            })
        
        runs_data.append(run_data)
    
    return pd.DataFrame(runs_data)


def find_best_params(df: pd.DataFrame) -> Dict:
    """최적 파라미터 찾기"""
    
    # 완료된 run 중 val_h_mean이 있는 것만 필터링
    valid_df = df[(df['state'] == 'finished') & (df['val_h_mean'].notna())].copy()
    
    if len(valid_df) == 0:
        print("완료된 run이 없습니다.")
        return None
    
    # H-Mean 기준 최고 성능 찾기
    best_idx = valid_df['val_h_mean'].idxmax()
    best_run = valid_df.loc[best_idx]
    
    print("\n" + "="*70)
    print("최적 파라미터 찾음!")
    print("="*70)
    print(f"\nRun Name: {best_run['run_name']}")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Val H-Mean: {best_run['val_h_mean']:.6f}")
    print(f"Val Precision: {best_run['val_precision']:.6f}")
    print(f"Val Recall: {best_run['val_recall']:.6f}")
    
    print("\n📊 최적 파라미터:")
    print("-" * 70)
    print(f"Learning Rate (lr): {best_run['lr']:.8f}")
    print(f"Weight Decay: {best_run['weight_decay']:.8f}")
    print(f"T_max (스케줄러): {int(best_run['T_max'])}")
    print(f"Detection Threshold (thresh): {best_run['thresh']:.4f}")
    print(f"Box Threshold (box_thresh): {best_run['box_thresh']:.4f}")
    print(f"Batch Size: {int(best_run['batch_size'])}")
    print(f"Max Epochs: {int(best_run['max_epochs'])}")
    print("="*70 + "\n")
    
    # 학습 명령어 생성
    train_cmd = f"""cd /data/ephemeral/home/baseline_code && \\
source /data/ephemeral/home/venv/bin/activate && \\
python runners/train.py \\
  preset=hrnet_w44_1024 \\
  exp_name=hrnet_w44_1024_sweep_optimized \\
  trainer.max_epochs={int(best_run['max_epochs'])} \\
  models.optimizer.lr={best_run['lr']:.8f} \\
  models.optimizer.weight_decay={best_run['weight_decay']:.8f} \\
  models.scheduler.T_max={int(best_run['T_max'])} \\
  models.head.postprocess.thresh={best_run['thresh']:.6f} \\
  models.head.postprocess.box_thresh={best_run['box_thresh']:.6f} \\
  datasets.batch_size={int(best_run['batch_size'])} \\
  wandb=True"""
    
    print("🚀 최적 파라미터로 학습 시작 명령어:")
    print("-" * 70)
    print(train_cmd)
    print("="*70 + "\n")
    
    return {
        'run_name': best_run['run_name'],
        'run_id': best_run['run_id'],
        'val_h_mean': best_run['val_h_mean'],
        'val_precision': best_run['val_precision'],
        'val_recall': best_run['val_recall'],
        'params': {
            'lr': best_run['lr'],
            'weight_decay': best_run['weight_decay'],
            'T_max': int(best_run['T_max']),
            'thresh': best_run['thresh'],
            'box_thresh': best_run['box_thresh'],
            'batch_size': int(best_run['batch_size']),
            'max_epochs': int(best_run['max_epochs']),
        },
        'train_cmd': train_cmd
    }


def print_sweep_summary(df: pd.DataFrame):
    """Sweep 요약 출력"""
    
    print("\n📈 Sweep 실행 요약")
    print("="*70)
    print(f"총 Run 수: {len(df)}")
    print(f"완료된 Run: {len(df[df['state'] == 'finished'])}")
    print(f"진행 중인 Run: {len(df[df['state'] == 'running'])}")
    print(f"실패한 Run: {len(df[df['state'] == 'failed'])}")
    
    valid_df = df[(df['state'] == 'finished') & (df['val_h_mean'].notna())]
    if len(valid_df) > 0:
        print(f"\n완료된 Run 성능:")
        print(f"  최고 H-Mean: {valid_df['val_h_mean'].max():.6f}")
        print(f"  평균 H-Mean: {valid_df['val_h_mean'].mean():.6f}")
        print(f"  최저 H-Mean: {valid_df['val_h_mean'].min():.6f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # 환경 변수 설정
    os.environ['WANDB_API_KEY'] = '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'
    
    entity = "fc_bootcamp"
    project = "ocr-receipt-detection"
    sweep_id = "2gdum3s9"
    
    print(f"\n🔍 WandB Sweep 결과 분석 중...")
    print(f"Sweep ID: {sweep_id}")
    print(f"프로젝트: {entity}/{project}\n")
    
    # 결과 수집
    try:
        results_df = fetch_sweep_results(entity, project, sweep_id)
        print_sweep_summary(results_df)
        
        # 최적 파라미터 찾기
        best_params = find_best_params(results_df)
        
        if best_params:
            # 결과 저장
            results_df.to_csv('/data/ephemeral/home/baseline_code/sweep_results.csv', index=False)
            print("✅ 결과가 sweep_results.csv에 저장되었습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
