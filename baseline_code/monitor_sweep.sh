#!/bin/bash
# Sweep 진행 상황 모니터링

echo "=== WandB Sweep 모니터링 ===" 
echo "Sweep ID: 2gdum3s9"
echo "프로젝트: fc_bootcamp/ocr-receipt-detection"
echo ""

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 학습 프로세스 상태:"
    ps aux | grep "train.py" | grep -v grep | wc -l
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 마지막 로그:"
    tail -3 /data/ephemeral/home/baseline_code/sweep_agent_v2.log 2>/dev/null | tail -1
    
    echo "---"
    sleep 60
done
