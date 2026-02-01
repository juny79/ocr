#!/bin/bash

# WandB Sweep 초기화 및 실행 스크립트
# Purpose: EfficientNet-B4 하이퍼파라미터 최적화

set -e

echo "========================================="
echo "WandB Sweep 초기화"
echo "========================================="
echo ""

# 설정
SWEEP_CONFIG="configs/sweep_efficientnet_b4.yaml"
PROJECT="fc_bootcamp/ocr-receipt-detection"
ENTITY="quriquri7"
NUM_RUNS=${1:-10}  # 기본값: 10회 실행

echo "📊 Sweep 설정"
echo "-----------------------------------------"
echo "Config File: ${SWEEP_CONFIG}"
echo "WandB Project: ${PROJECT}"
echo "WandB Entity: ${ENTITY}"
echo "Number of Runs: ${NUM_RUNS}"
echo ""

# WandB 로그인 확인
echo "🔐 WandB 인증"
echo "-----------------------------------------"
wandb login --relogin
echo ""

# Sweep 생성
echo "🚀 Sweep 생성 중..."
echo "-----------------------------------------"
SWEEP_ID=$(wandb sweep ${SWEEP_CONFIG} --project ${PROJECT} --entity ${ENTITY} 2>&1 | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Sweep 생성 실패"
    exit 1
fi

echo "✅ Sweep 생성 완료"
echo "Sweep ID: ${SWEEP_ID}"
echo ""

# Sweep 정보 저장
echo "${SWEEP_ID}" > sweep_id.txt
echo "Sweep ID가 sweep_id.txt에 저장되었습니다."
echo ""

# Sweep 링크
echo "🌐 WandB Sweep 링크"
echo "-----------------------------------------"
echo "https://wandb.ai/${ENTITY}/$(echo ${PROJECT} | tr '/' '-')/sweeps/$(basename ${SWEEP_ID})"
echo ""

# Sweep Agent 시작 안내
echo "========================================="
echo "Sweep Agent 시작 방법"
echo "========================================="
echo ""
echo "옵션 1: 자동 시작 (현재 터미널)"
echo "-----------------------------------------"
echo "wandb agent ${SWEEP_ID} --count ${NUM_RUNS}"
echo ""
echo "옵션 2: 백그라운드 실행"
echo "-----------------------------------------"
echo "nohup wandb agent ${SWEEP_ID} --count ${NUM_RUNS} > sweep_log.txt 2>&1 &"
echo ""
echo "옵션 3: 여러 에이전트 병렬 실행 (GPU 여러 개)"
echo "-----------------------------------------"
echo "# Terminal 1"
echo "CUDA_VISIBLE_DEVICES=0 wandb agent ${SWEEP_ID} --count 5 &"
echo "# Terminal 2"
echo "CUDA_VISIBLE_DEVICES=1 wandb agent ${SWEEP_ID} --count 5 &"
echo ""

# 자동 시작 여부 확인
read -p "지금 Sweep을 시작하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🏃 Sweep Agent 시작..."
    echo "========================================="
    wandb agent ${SWEEP_ID} --count ${NUM_RUNS}
else
    echo ""
    echo "ℹ️  나중에 다음 명령어로 시작하세요:"
    echo "wandb agent ${SWEEP_ID} --count ${NUM_RUNS}"
fi

echo ""
echo "========================================="
echo "Sweep 모니터링 명령어"
echo "========================================="
echo ""
echo "# 실행 중인 에이전트 확인"
echo "ps aux | grep 'wandb agent'"
echo ""
echo "# 로그 실시간 확인"
echo "tail -f sweep_log.txt"
echo ""
echo "# Sweep 중단"
echo "pkill -f 'wandb agent'"
echo ""
