#!/bin/bash

# EfficientNet-B4 단일 모델 학습 스크립트
# Purpose: EfficientNet-B4 백본으로 첫 번째 모델 학습 및 성능 검증

set -e

echo "========================================="
echo "EfficientNet-B4 단일 모델 학습 시작"
echo "========================================="
echo ""

# 환경 정보
echo "📊 시스템 정보"
echo "-----------------------------------------"
echo "날짜: $(date)"
echo "작업 디렉토리: $(pwd)"
echo "GPU 정보:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# 디스크 용량 확인
echo "💾 디스크 용량"
echo "-----------------------------------------"
df -h | grep -E "Filesystem|/data"
echo ""

# 학습 설정
PRESET="efficientnet_b4_aggressive"
EXP_NAME="efficientnet_b4_single"
EPOCHS=22
WANDB_ENABLED=false

echo "🔧 학습 설정"
echo "-----------------------------------------"
echo "Preset: ${PRESET}"
echo "Experiment Name: ${EXP_NAME}"
echo "Max Epochs: ${EPOCHS}"
echo "WandB Logging: ${WANDB_ENABLED}"
echo "Learning Rate: 0.0003"
echo "Weight Decay: 0.0001"
echo "Resolution: 960x960"
echo "Batch Size: 4"
echo ""

# 출력 디렉토리 생성
mkdir -p outputs/${EXP_NAME}/checkpoints
mkdir -p outputs/${EXP_NAME}/logs

# WandB 로그인 확인
if [ "$WANDB_ENABLED" = true ]; then
    echo "🌐 WandB 연결 확인"
    echo "-----------------------------------------"
    wandb login --relogin
    echo ""
fi

# 학습 시작 시간 기록
START_TIME=$(date +%s)
echo "⏱️  학습 시작: $(date)"
echo ""

# 학습 실행
python runners/train.py \
    preset=${PRESET} \
    exp_name=${EXP_NAME} \
    trainer.max_epochs=${EPOCHS} \
    wandb=${WANDB_ENABLED} \
    wandb_config.tags=['efficientnet_b4','single_model','baseline'] \
    wandb_config.notes='EfficientNet-B4 initial training for performance validation' \
    2>&1 | tee outputs/${EXP_NAME}/logs/training_$(date +%Y%m%d_%H%M%S).log

# 학습 종료 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================="
echo "학습 완료"
echo "========================================="
echo ""
echo "⏱️  소요 시간: ${HOURS}시간 ${MINUTES}분"
echo ""

# 체크포인트 확인
echo "📦 생성된 체크포인트"
echo "-----------------------------------------"
if [ -d "outputs/${EXP_NAME}/checkpoints" ]; then
    ls -lh outputs/${EXP_NAME}/checkpoints/*.ckpt 2>/dev/null || echo "체크포인트 파일을 찾을 수 없습니다."
else
    echo "체크포인트 디렉토리가 생성되지 않았습니다."
fi
echo ""

# WandB 링크 출력
if [ "$WANDB_ENABLED" = true ]; then
    echo "🌐 WandB 링크"
    echo "-----------------------------------------"
    echo "Project: https://wandb.ai/quriquri7/fc_bootcamp/ocr-receipt-detection"
    echo "Run: ${EXP_NAME}"
    echo ""
fi

# 다음 단계 안내
echo "📋 다음 단계"
echo "-----------------------------------------"
echo "1. WandB에서 validation H-Mean 확인"
echo "2. bash scripts/predict_efficientnet_b4.sh 실행"
echo "3. 성능 평가 후 다음 전략 결정:"
echo "   - ≥96.5%: 5-Fold 학습 진행"
echo "   - 96.3-96.5%: ResNet50과 2-way 앙상블"
echo "   - <96.3%: 하이퍼파라미터 튜닝 (sweep)"
echo ""
