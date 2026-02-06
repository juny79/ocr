#!/bin/bash

# WandB Sweep 실행 스크립트 for HRNet-W44 1280x1280

set -e

cd /data/ephemeral/home/baseline_code

echo "================================"
echo "WandB Sweep 설정 시작"
echo "================================"
echo ""
echo "Sweep 파일: sweep_hrnet_w44_1280.yaml"
echo ""
echo "탐색 파라미터:"
echo "  - Learning Rate (lr): exp(-11.5) ~ exp(-8.5)"
echo "    현재값: 0.00045"
echo ""
echo "  - Weight Decay: exp(-12) ~ exp(-9)"
echo "    현재값: 0.00006"
echo ""
echo "  - T_max: [15, 18, 20, 25]"
echo "    현재값: 20"
echo ""
echo "  - eta_min: exp(-13) ~ exp(-10)"
echo "    현재값: 0.000008"
echo ""
echo "조기 종료: Hyperband (5 epoch 이후)"
echo ""
echo "================================"
echo ""

# Sweep 초기화
echo "1️⃣  Sweep 초기화 중..."
SWEEP_ID=$(wandb sweep sweep_hrnet_w44_1280.yaml --project hrnet-w44-1280-sweep --entity juny79 2>&1 | grep -oP 'Run sweep agent with: wandb agent.*' | sed 's/Run sweep agent with: //' | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Sweep ID를 얻지 못했습니다. 수동으로 다음 명령어를 실행하세요:"
    echo "  wandb sweep sweep_hrnet_w44_1280.yaml --project hrnet-w44-1280-sweep --entity juny79"
    exit 1
fi

echo "✅ Sweep ID: $SWEEP_ID"
echo ""
echo "2️⃣  에이전트 실행 중 (최대 8개 parallel runs)..."
echo ""

# Sweep 에이전트 실행 (최대 8개 병렬 실행)
wandb agent --count 8 "$SWEEP_ID"

echo ""
echo "================================"
echo "Sweep 완료!"
echo "================================"
echo ""
echo "결과 확인: https://wandb.ai/juny79/hrnet-w44-1280-sweep"
echo ""
