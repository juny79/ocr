#!/usr/bin/env python
"""
WandB Sweep을 위한 학습 래퍼 스크립트
sweep_config.yaml의 파라미터를 받아서 Hydra 형식으로 변환합니다.
"""

import os
import sys
import wandb
from pathlib import Path
import subprocess

def run_sweep_trial():
    """WandB config에서 파라미터를 읽고 학습 실행"""
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:128",
    )
    
    # WandB 초기화
    wandb.init()
    
    # WandB config에서 파라미터 추출
    config = wandb.config
    
    # Hydra 파라미터로 변환
    batch_size = int(config.batch_size)
    if batch_size > 1:
        batch_size = 1

    hydra_args = [
        "python", "runners/train.py",
        f"models.optimizer.lr={config.lr}",
        f"models.optimizer.weight_decay={config.weight_decay}",
        f"models.scheduler.T_max={int(config.T_max)}",
        f"models.head.postprocess.thresh={config.thresh}",
        f"models.head.postprocess.box_thresh={config.box_thresh}",
        f"dataloaders.train_dataloader.batch_size={batch_size}",
        f"dataloaders.val_dataloader.batch_size={batch_size}",
        f"dataloaders.test_dataloader.batch_size={batch_size}",
        f"preset={config.preset}",
        f"trainer.max_epochs={int(config.max_epochs)}",
        f"exp_name=sweep_{wandb.run.name}",
        "wandb=True",
    ]
    
    print(f"Running command: {' '.join(hydra_args)}")
    print(f"Current directory: {os.getcwd()}")
    
    # 학습 실행
    result = subprocess.run(hydra_args, cwd="/data/ephemeral/home/baseline_code")
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    run_sweep_trial()
