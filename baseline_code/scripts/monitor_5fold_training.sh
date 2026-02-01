#!/bin/bash

# Monitor 5-Fold Training Progress
# Usage: bash scripts/monitor_5fold_training.sh

BASELINE_DIR="/data/ephemeral/home/baseline_code"
LOG_FILE="$BASELINE_DIR/logs/train_folds_234_v2.log"

while true; do
    clear
    echo "========================================"
    echo "5-Fold Training Monitor"
    echo "$(date)"
    echo "========================================"
    echo ""
    
    # Process status
    echo "=== 활성 프로세스 ==="
    ps aux | grep "train.py" | grep -v grep | wc -l | xargs echo "Train.py 프로세스:"
    echo ""
    
    # GPU utilization
    echo "=== GPU 사용률 ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi 사용 불가"
    echo ""
    
    # Training progress
    echo "=== 훈련 진행 상황 ==="
    if [ -f "$LOG_FILE" ]; then
        # Current fold
        current_fold=$(tail -100 "$LOG_FILE" | grep "Training Fold" | tail -1)
        if [ -n "$current_fold" ]; then
            echo "$current_fold"
        fi
        
        # Latest epoch info
        latest_epoch=$(tail -50 "$LOG_FILE" | grep -E "Epoch [0-9]+" | tail -1)
        if [ -n "$latest_epoch" ]; then
            echo "$latest_epoch"
        fi
        
        # Validation scores
        echo ""
        echo "최근 Validation 점수:"
        tail -200 "$LOG_FILE" | grep -E "val/hmean|val/precision|val/recall" | tail -3
    else
        echo "로그 파일 없음"
    fi
    echo ""
    
    # Checkpoint status
    echo "=== 체크포인트 현황 ==="
    for fold in 0 1 2 3 4; do
        ckpt_dir="$BASELINE_DIR/outputs/resnet50_fold${fold}/checkpoints"
        if [ -d "$ckpt_dir" ]; then
            count=$(ls "$ckpt_dir"/*.ckpt 2>/dev/null | wc -l)
            echo "  Fold $fold: $count 체크포인트"
        else
            echo "  Fold $fold: 디렉토리 없음"
        fi
    done
    echo ""
    
    # Estimated completion
    echo "=== 예상 완료 시간 ==="
    echo "  Fold당 훈련 시간: ~1.5-2시간"
    echo "  남은 Fold: Fold 2, 3, 4"
    echo "  예상 총 시간: ~5-6시간"
    echo ""
    
    echo "========================================"
    echo "Press Ctrl+C to exit"
    echo "자동 업데이트: 30초마다"
    echo "========================================"
    
    sleep 30
done
