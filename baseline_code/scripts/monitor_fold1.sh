#!/bin/bash

echo "========================================"
echo "Fold 1 Training Monitor"
echo "Time: $(date)"
echo "========================================"
echo ""

# 프로세스 확인
PROCESS_COUNT=$(ps aux | grep "python.*train.py.*fold1" | grep -v grep | wc -l)
echo "Active training processes: $PROCESS_COUNT"

if [ $PROCESS_COUNT -eq 0 ]; then
    echo "Status: ✅ Training completed or not started"
    echo ""
    
    # 체크포인트 확인
    if [ -d "/data/ephemeral/home/baseline_code/outputs/resnet50_fold1/checkpoints" ]; then
        CKPT_COUNT=$(ls /data/ephemeral/home/baseline_code/outputs/resnet50_fold1/checkpoints/*.ckpt 2>/dev/null | wc -l)
        echo "Checkpoints found: $CKPT_COUNT"
        
        if [ $CKPT_COUNT -gt 0 ]; then
            echo ""
            echo "Latest checkpoint:"
            ls -lth /data/ephemeral/home/baseline_code/outputs/resnet50_fold1/checkpoints/*.ckpt | head -1
            echo ""
            echo "✅ Ready for ensemble!"
            echo "Run: bash scripts/run_2fold_ensemble.sh"
        fi
    fi
else
    echo "Status: 🔄 Training in progress..."
    echo ""
    
    # 로그 확인
    if [ -f "/data/ephemeral/home/baseline_code/logs/resnet50_fold1_master.log" ]; then
        echo "Recent training log (last 20 lines):"
        echo "--------------------------------"
        tail -20 /data/ephemeral/home/baseline_code/logs/resnet50_fold1_master.log
    fi
fi

echo ""
echo "========================================"
