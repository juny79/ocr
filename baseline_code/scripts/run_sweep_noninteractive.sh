#!/bin/bash

# WandB Sweep - Non-Interactive Mode
# Learning Rate Optimization without interactive login

set -e

echo "========================================="
echo "WandB Sweep - Learning Rate 최적화"
echo "========================================="
echo ""

# Check WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ WANDB_API_KEY 환경변수가 설정되지 않았습니다."
    echo ""
    echo "다음 명령어로 API Key를 설정하세요:"
    echo "export WANDB_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✅ WANDB_API_KEY 확인됨"
echo ""

# Sweep 설정
SWEEP_CONFIG="configs/sweep_efficientnet_b4_lr_optimized.yaml"
NUM_RUNS=${1:-12}

echo "📋 Sweep 정보"
echo "-----------------------------------------"
echo "Base 성능: 96.37% (Postprocessing 최적화 완료)"
echo "목표: 96.50%+"
echo "전략: Learning Rate + Weight Decay 최적화"
echo ""
echo "Config: ${SWEEP_CONFIG}"
echo "실행 횟수: ${NUM_RUNS}회"
echo "예상 소요 시간: $((NUM_RUNS * 120 / 60))시간"
echo ""

# Set WandB mode
export WANDB_MODE=online
export WANDB_SILENT=true

# Sweep 초기화 (login 없이)
echo "🚀 Sweep 초기화 중..."
SWEEP_OUTPUT=$(wandb sweep ${SWEEP_CONFIG} 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^\s]+' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Sweep 초기화 실패"
    echo "출력:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

echo "✅ Sweep ID: ${SWEEP_ID}"
echo ""

# 로그 디렉토리 생성
LOG_DIR="logs/sweep_lr_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/sweep.log"

echo "📝 로그: ${LOG_FILE}"
echo ""

# Sweep 실행
echo "========================================="
echo "WandB Sweep 실행 중..."
echo "========================================="
echo ""
echo "진행상황은 WandB 대시보드에서 확인:"
echo "https://wandb.ai"
echo ""

# Background 실행 여부
if [ "$2" == "bg" ] || [ "$2" == "background" ]; then
    echo "🔄 Background 모드로 실행합니다..."
    nohup wandb agent --count ${NUM_RUNS} ${SWEEP_ID} > ${LOG_FILE} 2>&1 &
    AGENT_PID=$!
    echo "✅ Agent PID: ${AGENT_PID}"
    echo ""
    echo "중지: kill ${AGENT_PID}"
    echo "로그: tail -f ${LOG_FILE}"
    echo ""
    echo "약 $((NUM_RUNS * 120 / 60))시간 후 완료 예상"
else
    echo "🔄 Foreground 모드로 실행합니다..."
    echo "(Ctrl+C로 중지 가능)"
    echo ""
    wandb agent --count ${NUM_RUNS} ${SWEEP_ID} 2>&1 | tee ${LOG_FILE}
fi

echo ""
echo "========================================="
echo "✅ Sweep 시작됨!"
echo "========================================="
echo ""
