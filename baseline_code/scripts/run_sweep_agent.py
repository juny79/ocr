#!/usr/bin/env python3
"""
WandB Sweep Agent를 실행하는 스크립트
Sweep ID를 입력받아 K-Fold 학습 실행
"""
import wandb
import subprocess
import sys

# Sweep ID 입력
if len(sys.argv) > 1:
    sweep_id = sys.argv[1]
else:
    sweep_id = input("Sweep ID 입력: ").strip()

print(f"🚀 Sweep Agent 시작: {sweep_id}")

# Agent 실행
def train():
    # WandB config 가져오기
    run = wandb.init()
    config = wandb.config
    
    # 학습 명령 구성
    cmd = [
        "python", "runners/train.py",
        "preset=hrnet_w44_1024",
        "trainer.max_epochs=13",
        "datasets.train_dataset.annotation_path=/data/ephemeral/home/data/datasets/jsons/train_augmented_full.json",
        "wandb=true"
    ]
    
    # Config에서 파라미터 추가
    for key, value in config.items():
        cmd.append(f"{key}={value}")
    
    print(f"📌 실행 명령: {' '.join(cmd)}")
    
    # 학습 실행
    result = subprocess.run(cmd, cwd="/data/ephemeral/home/baseline_code")
    
    if result.returncode != 0:
        print(f"❌ 학습 실패: exit code {result.returncode}")
        sys.exit(1)

# Sweep agent 시작
wandb.agent(sweep_id, function=train, count=30)
