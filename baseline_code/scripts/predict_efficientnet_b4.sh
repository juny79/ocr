#!/bin/bash

# EfficientNet-B4 Prediction 생성 스크립트
# Purpose: 학습된 EfficientNet-B4 모델로 예측 생성

set -e

echo "========================================="
echo "EfficientNet-B4 Prediction 생성"
echo "========================================="
echo ""

# 설정
EXP_NAME="efficientnet_b4_single"
CHECKPOINT_DIR="outputs/${EXP_NAME}/checkpoints"
OUTPUT_DIR="outputs/${EXP_NAME}_predict"

# 체크포인트 확인
echo "📦 체크포인트 확인"
echo "-----------------------------------------"
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "❌ 체크포인트 디렉토리를 찾을 수 없습니다: ${CHECKPOINT_DIR}"
    exit 1
fi

# 최신 체크포인트 찾기
LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*.ckpt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ 체크포인트 파일을 찾을 수 없습니다."
    exit 1
fi

echo "체크포인트: ${LATEST_CHECKPOINT}"
echo "크기: $(du -h ${LATEST_CHECKPOINT} | cut -f1)"
echo ""

# Hydra 에스케이핑 처리
CHECKPOINT_ESCAPED=$(echo $LATEST_CHECKPOINT | sed 's/=/\\=/g')
echo "에스케이프된 경로: ${CHECKPOINT_ESCAPED}"
echo ""

# 출력 디렉토리 생성
mkdir -p ${OUTPUT_DIR}/submissions
mkdir -p ${OUTPUT_DIR}/logs

# Prediction 실행
echo "🚀 Prediction 시작"
echo "-----------------------------------------"
echo "시작 시간: $(date)"
echo ""

START_TIME=$(date +%s)

python runners/predict.py \
    preset=efficientnet_b4_aggressive \
    exp_name=${EXP_NAME}_predict \
    checkpoint_path=${CHECKPOINT_ESCAPED} \
    2>&1 | tee ${OUTPUT_DIR}/logs/predict_$(date +%Y%m%d_%H%M%S).log

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ Prediction 완료"
echo "소요 시간: ${DURATION}초"
echo ""

# 결과 파일 확인
echo "📊 생성된 파일"
echo "-----------------------------------------"
LATEST_JSON=$(ls -t ${OUTPUT_DIR}/submissions/*.json 2>/dev/null | head -1)
if [ -n "$LATEST_JSON" ]; then
    echo "JSON: ${LATEST_JSON}"
    echo "크기: $(du -h ${LATEST_JSON} | cut -f1)"
    
    # JSON을 CSV로 변환
    echo ""
    echo "🔄 CSV 변환 중..."
    python ocr/utils/convert_submission.py ${LATEST_JSON}
    
    LATEST_CSV="${LATEST_JSON%.json}.csv"
    if [ -f "$LATEST_CSV" ]; then
        echo "✅ CSV: ${LATEST_CSV}"
        echo "크기: $(du -h ${LATEST_CSV} | cut -f1)"
    fi
else
    echo "⚠️ 결과 파일을 찾을 수 없습니다."
fi
echo ""

# 다음 단계 안내
echo "========================================="
echo "다음 단계"
echo "========================================="
echo ""
echo "1. 리더보드에 ${LATEST_CSV} 제출"
echo ""
echo "2. 성능 평가:"
echo "   - ≥96.5%: 5-Fold 학습 진행"
echo "     → bash scripts/train_efficientnet_b4_5fold.sh"
echo ""
echo "   - 96.3-96.5%: ResNet50과 2-way 앙상블"
echo "     → python scripts/ensemble_resnet_effnet.py"
echo ""
echo "   - <96.3%: 하이퍼파라미터 튜닝"
echo "     → bash scripts/start_sweep.sh"
echo ""
