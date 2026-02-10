#!/usr/bin/env python3
"""
WandB Sweep을 Python API로 직접 생성하고 실행
"""
import wandb
import os

# .env에서 API key 로드
from pathlib import Path
env_file = Path("/data/ephemeral/home/baseline_code/.env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.startswith("WANDB_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                os.environ["WANDB_API_KEY"] = api_key

# WandB 로그인
wandb.login()

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/hmean',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 8,
        'eta': 2
    },
    'parameters': {
        # Learning Rate
        'models.optimizer.lr': {
            'distribution': 'log_uniform_values',
            'min': 0.0008,
            'max': 0.0020
        },
        # Weight Decay
        'models.optimizer.weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.0008
        },
        # Scheduler T_max
        'models.scheduler.T_max': {
            'values': [8, 10, 12, 15]
        },
        # Scheduler eta_min
        'models.scheduler.eta_min': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-5
        },
        # Batch Size
        'dataloader.batch_size': {
            'values': [2, 3, 4]
        },
        # Postprocessing - Threshold
        'models.head.postprocess.thresh': {
            'distribution': 'uniform',
            'min': 0.18,
            'max': 0.28
        },
        # Postprocessing - Box Threshold
        'models.head.postprocess.box_thresh': {
            'distribution': 'uniform',
            'min': 0.35,
            'max': 0.50
        },
        # Loss weights
        'models.loss.negative_ratio': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 4.0
        },
        'models.loss.prob_map_loss_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0
        },
        'models.loss.thresh_map_loss_weight': {
            'distribution': 'uniform',
            'min': 6.0,
            'max': 10.0
        },
        'models.loss.binary_map_loss_weight': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.5
        },
        # CollateFN parameters
        'collate_fn.shrink_ratio': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.5
        },
        'collate_fn.thresh_min': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.4
        },
        'collate_fn.thresh_max': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 0.85
        }
    }
}

# Sweep 생성
print("🔄 WandB Sweep 생성 중...")
sweep_id = wandb.sweep(sweep_config, project="ocr-receipt-detection")
print(f"✅ Sweep 생성 완료!")
print(f"📊 Sweep ID: {sweep_id}")
print(f"\n🚀 Agent 시작 명령:")
print(f"wandb agent {sweep_id}")
print(f"\n또는:")
print(f"wandb agent quriquri7/ocr-receipt-detection/{sweep_id}")
