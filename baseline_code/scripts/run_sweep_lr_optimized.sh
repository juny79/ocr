#!/bin/bash

# WandB Sweep - Learning Rate Optimization
# Base: 96.37% (thresh=0.28, box_thresh=0.25)
# Goal: 96.50%+

set -e

echo "========================================="
echo "WandB Sweep - Learning Rate 최적화"
echo "========================================="
echo ""

# WandB 설정 확인
echo "📋 Sweep 정보"
echo "-----------------------------------------"
echo "Base 성능: 96.37% (Postprocessing 최적화 완료)"
echo "목표: 96.50%+"
echo "전략: Learning Rate + Weight Decay 최적화"
echo ""
echo "고정 파라미터:"
echo "  - thresh: 0.28 (최적값)"
echo "  - box_thresh: 0.25 (최적값)"
echo "  - max_candidates: 600"
echo ""
echo "탐색 파라미터:"
echo "  - Learning Rate: 0.00025 - 0.0006"
echo "  - Weight Decay: 0.00005 - 0.0005"
echo "  - T_Max: 20, 22, 24"
echo "  - eta_min: 0.000005 - 0.00005"
echo ""

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY 환경변수가 설정되지 않았습니다."
    echo ""
    echo "WandB API Key를 입력하세요 (https://wandb.ai/authorize):"
    read -s WANDB_API_KEY
    export WANDB_API_KEY=$WANDB_API_KEY
    echo ""
fi

# WandB 로그인
echo "🔐 WandB 로그인 확인 중..."
wandb login --relogin <<< "$WANDB_API_KEY" 2>&1 | grep -q "Successfully logged in" && echo "✅ 로그인 성공!" || echo "❌ 로그인 실패"
echo ""

# Sweep 설정
SWEEP_CONFIG="configs/sweep_efficientnet_b4_lr_optimized.yaml"
NUM_RUNS=${1:-12}  # 기본 12회 (LR에 집중)

echo "📊 Sweep 실행 계획"
echo "-----------------------------------------"
echo "Config: ${SWEEP_CONFIG}"
echo "실행 횟수: ${NUM_RUNS}회"
echo "예상 소요 시간: $((NUM_RUNS * 120 / 60))시간"
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

# 로그 파일 설정
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
echo "https://wandb.ai/[YOUR-USERNAME]/efficientnet_b4_sweep_lr/sweeps/${SWEEP_ID##*/}"
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
else
    echo "🔄 Interactive 모드로 실행합니다..."
    wandb agent --count ${NUM_RUNS} ${SWEEP_ID} 2>&1 | tee ${LOG_FILE}
fi

echo ""
echo "========================================="
echo "✅ Sweep 완료!"
echo "========================================="
echo ""
echo "📊 다음 단계:"
echo "1. WandB 대시보드에서 최고 성능 run 확인"
echo "2. 최적 하이퍼파라미터 확인 (목표: 96.50%+)"
echo "3. 최적 설정으로 5-Fold 학습 진행"
echo ""
