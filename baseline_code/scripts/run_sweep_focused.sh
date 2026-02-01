#!/bin/bash

# WandB Sweep - EfficientNet-B4 Focused Optimization
# Purpose: 96.00% → 96.50% H-Mean 개선

set -e

echo "========================================="
echo "WandB Sweep - EfficientNet-B4 최적화"
echo "========================================="
echo ""

# WandB 설정 확인
echo "📋 WandB 설정 확인"
echo "-----------------------------------------"

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY 환경변수가 설정되지 않았습니다."
    echo ""
    echo "WandB API Key를 입력하세요 (https://wandb.ai/authorize):"
    read -s WANDB_API_KEY
    export WANDB_API_KEY=$WANDB_API_KEY
    echo ""
fi

# WandB 로그인 확인
echo "🔐 WandB 로그인 확인 중..."
wandb login --relogin <<< "$WANDB_API_KEY" 2>&1 | grep -q "Successfully logged in" && echo "✅ 로그인 성공!" || echo "❌ 로그인 실패"
echo ""

# Sweep 설정
SWEEP_CONFIG="configs/sweep_efficientnet_b4_focused.yaml"
NUM_RUNS=${1:-15}  # 기본 15회

echo "📊 Sweep 설정"
echo "-----------------------------------------"
echo "Config: ${SWEEP_CONFIG}"
echo "실행 횟수: ${NUM_RUNS}회"
echo "예상 소요 시간: $((NUM_RUNS * 25 / 60))시간"
echo ""

# Sweep 초기화
echo "🚀 Sweep 초기화 중..."
SWEEP_ID=$(wandb sweep ${SWEEP_CONFIG} 2>&1 | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Sweep 초기화 실패"
    exit 1
fi

echo "✅ Sweep ID: ${SWEEP_ID}"
echo ""

# Sweep Agent 실행
echo "🤖 Sweep Agent 시작"
echo "-----------------------------------------"
echo "시작 시간: $(date)"
echo ""

# 로그 파일 설정
LOG_DIR="logs/sweep_$(date +%Y%m%d_%H%M%S)"
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
echo "https://wandb.ai/[YOUR-USERNAME]/efficientnet_b4_sweep_focused/sweeps/${SWEEP_ID##*/}"
echo ""
echo "로그 실시간 확인:"
echo "tail -f ${LOG_FILE}"
echo ""

# Background 실행 여부 확인
if [ "$2" == "bg" ] || [ "$2" == "background" ]; then
    echo "🔄 Background 모드로 실행합니다..."
    nohup wandb agent --count ${NUM_RUNS} ${SWEEP_ID} > ${LOG_FILE} 2>&1 &
    AGENT_PID=$!
    echo "✅ Agent PID: ${AGENT_PID}"
    echo ""
    echo "중지하려면: kill ${AGENT_PID}"
    echo "로그 확인: tail -f ${LOG_FILE}"
else
    echo "🔄 Interactive 모드로 실행합니다..."
    echo "(Ctrl+C로 중지 가능, 이어서 계속하려면 같은 명령어 재실행)"
    echo ""
    wandb agent --count ${NUM_RUNS} ${SWEEP_ID} 2>&1 | tee ${LOG_FILE}
fi

echo ""
echo "========================================="
echo "✅ Sweep 완료!"
echo "========================================="
echo ""
echo "📊 결과 확인:"
echo "1. WandB 대시보드에서 최고 성능 run 확인"
echo "2. 최적 하이퍼파라미터 복사"
echo "3. configs/preset/efficientnet_b4_optimal.yaml 생성"
echo "4. 최적 설정으로 재학습"
echo ""
