#!/bin/bash
# WandB Sweep 실행 스크립트

set -e

cd /data/ephemeral/home/baseline_code
source /data/ephemeral/home/venv/bin/activate

echo "🚀 WandB Sweep 초기화 중..."
echo ""

# Sweep 생성
SWEEP_ID=$(wandb sweep configs/sweep_hrnet_w44_optimized_1024.yaml 2>&1 | grep -oP 'wandb agent \K[^ ]+')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Sweep 생성 실패"
    exit 1
fi

echo "✅ Sweep 생성 완료: $SWEEP_ID"
echo ""
echo "📊 Sweep 실행 명령:"
echo "   wandb agent $SWEEP_ID"
echo ""
echo "🔗 Sweep 대시보드:"
echo "   https://wandb.ai/quriquri7/ocr-receipt-detection/sweeps/$SWEEP_ID"
echo ""

# Agent 실행 (옵션)
read -p "🤔 Sweep agent를 바로 시작하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🏃 Agent 시작..."
    wandb agent "$SWEEP_ID"
else
    echo "⏸️  나중에 실행하려면:"
    echo "   wandb agent $SWEEP_ID"
fi
