#!/usr/bin/env python3
"""
WandB Sweep 대신 직접 하이퍼파라미터 최적화 실행
Random Search로 파라미터 조합을 테스트
"""
import os
import sys
import subprocess
import random
import json
from pathlib import Path

# 파라미터 범위 정의 (sweep_simple.yaml 기반)
param_ranges = {
    'models.optimizer.lr': (0.0008, 0.0020),
    'models.optimizer.weight_decay': (0.0001, 0.0008),
    'models.scheduler.T_max': [8, 10, 12],
    'models.head.postprocess.thresh': (0.20, 0.26),
    'models.head.postprocess.box_thresh': (0.38, 0.48),
}

# 고정 파라미터
fixed_params = {
    'preset': 'hrnet_w44_1024',
    'trainer.max_epochs': 13,
    'datasets.train_dataset.annotation_path': '/data/ephemeral/home/data/datasets/jsons/train_augmented_full.json',
}

def generate_random_params():
    """랜덤 파라미터 조합 생성"""
    params = {}
    
    # 연속형 파라미터
    params['lr'] = random.uniform(*param_ranges['models.optimizer.lr'])
    params['weight_decay'] = random.uniform(*param_ranges['models.optimizer.weight_decay'])
    params['thresh'] = random.uniform(*param_ranges['models.head.postprocess.thresh'])
    params['box_thresh'] = random.uniform(*param_ranges['models.head.postprocess.box_thresh'])
    
    # 이산형 파라미터
    params['T_max'] = random.choice(param_ranges['models.scheduler.T_max'])
    
    return params

def run_training(run_id, params):
    """학습 실행"""
    exp_name = f"hyperparam_search_run_{run_id}"
    
    # 명령어 구성
    cmd = [
        'python', 'runners/train.py',
        f"preset={fixed_params['preset']}",
        f"trainer.max_epochs={fixed_params['trainer.max_epochs']}",
        f"datasets.train_dataset.annotation_path={fixed_params['datasets.train_dataset.annotation_path']}",
        f"models.optimizer.lr={params['lr']:.6f}",
        f"models.optimizer.weight_decay={params['weight_decay']:.6f}",
        f"models.scheduler.T_max={params['T_max']}",
        f"models.head.postprocess.thresh={params['thresh']:.4f}",
        f"models.head.postprocess.box_thresh={params['box_thresh']:.4f}",
        f"exp_name={exp_name}",
    ]
    
    print(f"\n{'='*80}")
    print(f"🚀 Run #{run_id} 시작")
    print(f"{'='*80}")
    print(f"📊 파라미터:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    print(f"{'='*80}\n")
    
    # 학습 실행
    result = subprocess.run(
        cmd,
        cwd='/data/ephemeral/home/baseline_code',
        env={**os.environ, 'WANDB_PROJECT': 'hrnet-w44-1024-kfold'}
    )
    
    return result.returncode == 0

def main():
    """메인 함수"""
    num_runs = 10  # 실행할 총 횟수
    
    print("="*80)
    print("🔍 하이퍼파라미터 탐색 시작")
    print("="*80)
    print(f"총 실행 횟수: {num_runs}")
    print(f"탐색 파라미터: {list(param_ranges.keys())}")
    print("="*80)
    
    results = []
    
    for run_id in range(1, num_runs + 1):
        # 랜덤 파라미터 생성
        params = generate_random_params()
        
        # 학습 실행
        success = run_training(run_id, params)
        
        # 결과 저장
        results.append({
            'run_id': run_id,
            'params': params,
            'success': success
        })
        
        # 결과 저장
        results_file = Path('/data/ephemeral/home/baseline_code/hyperparam_search_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ Run #{run_id} 완료 (성공: {success})")
        print(f"📝 결과 저장: {results_file}")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("🎉 모든 하이퍼파라미터 탐색 완료!")
    print("="*80)
    print(f"성공한 실행: {sum(1 for r in results if r['success'])}/{num_runs}")
    print(f"결과 파일: {results_file}")
    print("="*80)

if __name__ == '__main__':
    main()
