#!/bin/bash

echo "================================"
echo "ResNet50 Training Monitor"
echo "================================"
echo ""

# 프로세스 상태 확인
PROCESS_COUNT=$(ps aux | grep "python.*train.py" | grep resnet50 | grep -v grep | wc -l)
echo "Active training processes: $PROCESS_COUNT"
echo ""

# 최근 로그 출력
echo "Recent training log (last 30 lines):"
echo "--------------------------------"
tail -30 /data/ephemeral/home/baseline_code/logs/resnet50_fold0_master.log
echo ""
echo "================================"
echo "Monitoring script completed"
echo "================================"
