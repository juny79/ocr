#!/bin/bash

# Smart Postprocessing Optimization
# Purpose: 96.00% → 96.50% with minimal trials
# Strategy: Focused search on thresh/box_thresh

set -e

echo "========================================="
echo "🎯 Smart Postprocessing 최적화"
echo "========================================="
echo ""

# 설정
BASE_PRESET="efficientnet_b4_aggressive"
CHECKPOINT="outputs/efficientnet_b4_single/checkpoints/epoch=15-step=13088.ckpt"
OUTPUT_DIR="outputs/efficientnet_b4_postproc_optim"

mkdir -p ${OUTPUT_DIR}/submissions

echo "📊 현재 성능"
echo "-----------------------------------------"
echo "H-Mean:    96.00%"
echo "Precision: 96.27%"
echo "Recall:    95.98%"
echo ""
echo "🎯 목표: Precision↑ (96.27% → 97.0%+)"
echo "전략: thresh/box_thresh 증가로 False Positive 감소"
echo ""

echo "📋 최적화 전략 (9회 시도)"
echo "-----------------------------------------"
echo ""
echo "Phase 1: thresh 증가 (3회)"
echo "  1. thresh=0.24, box_thresh=0.25"
echo "  2. thresh=0.26, box_thresh=0.25"
echo "  3. thresh=0.28, box_thresh=0.25"
echo ""
echo "Phase 2: box_thresh 증가 (3회)"
echo "  4. thresh=0.24, box_thresh=0.28"
echo "  5. thresh=0.24, box_thresh=0.30"
echo "  6. thresh=0.26, box_thresh=0.28"
echo ""
echo "Phase 3: 조합 최적화 (3회)"
echo "  7. thresh=0.25, box_thresh=0.27"
echo "  8. thresh=0.23, box_thresh=0.26"
echo "  9. thresh=0.27, box_thresh=0.26"
echo ""
echo "예상 소요 시간: ~5분"
echo ""

# 실행 확인
read -p "9개 제출 파일을 생성하시겠습니까? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 1
fi

echo ""
echo "========================================="
echo "🚀 Postprocessing 최적화 시작"
echo "========================================="
echo ""

# 파라미터 조합 배열
declare -a CONFIGS=(
    "0.24 0.25 600"  # 1. thresh↑ (light)
    "0.26 0.25 600"  # 2. thresh↑ (medium)
    "0.28 0.25 600"  # 3. thresh↑ (heavy)
    "0.24 0.28 600"  # 4. box_thresh↑ (light)
    "0.24 0.30 600"  # 5. box_thresh↑ (medium)
    "0.26 0.28 600"  # 6. both↑
    "0.25 0.27 600"  # 7. balanced↑
    "0.23 0.26 600"  # 8. conservative
    "0.27 0.26 600"  # 9. aggressive
)

RUN_ID=1

for CONFIG in "${CONFIGS[@]}"; do
    read THRESH BOX_THRESH MAX_CAND <<< "$CONFIG"
    
    echo "[$RUN_ID/9] thresh=$THRESH, box_thresh=$BOX_THRESH, max_cand=$MAX_CAND"
    
    # 예측 실행
    python runners/predict.py \
        preset=${BASE_PRESET} \
        exp_name=postproc_optim_${RUN_ID} \
        checkpoint_path=${CHECKPOINT} \
        models.head.thresh=${THRESH} \
        models.head.box_thresh=${BOX_THRESH} \
        models.head.max_candidates=${MAX_CAND} \
        > ${OUTPUT_DIR}/log_${RUN_ID}.log 2>&1
    
    # JSON 파일 찾기
    JSON_FILE=$(ls -t outputs/postproc_optim_${RUN_ID}/submissions/*.json 2>/dev/null | head -1)
    
    if [ -n "$JSON_FILE" ]; then
        # CSV 변환
        CSV_FILE="${OUTPUT_DIR}/submissions/submission_t${THRESH}_b${BOX_THRESH}.csv"
        python ocr/utils/convert_submission.py \
            -J ${JSON_FILE} \
            -O ${CSV_FILE}
        
        echo "  ✅ ${CSV_FILE}"
    else
        echo "  ❌ 실패"
    fi
    
    RUN_ID=$((RUN_ID + 1))
    echo ""
done

echo "========================================="
echo "✅ 최적화 완료!"
echo "========================================="
echo ""
echo "📦 생성된 제출 파일 (9개)"
echo "-----------------------------------------"
ls -lh ${OUTPUT_DIR}/submissions/*.csv
echo ""
echo "📋 리더보드 제출 가이드"
echo "-----------------------------------------"
echo ""
echo "1. 9개 파일을 모두 리더보드에 제출"
echo ""
echo "2. 최고 성능 파라미터 확인"
echo "   (예: thresh=0.25, box_thresh=0.27 → 96.45%)"
echo ""
echo "3. 최적 설정으로 config 업데이트:"
echo "   configs/preset/efficientnet_b4_optimal.yaml"
echo ""
echo "4. 재학습 또는 5-Fold 진행"
echo ""
echo "🎯 기대 효과"
echo "-----------------------------------------"
echo "• thresh↑ → Precision↑, Recall↓"
echo "• box_thresh↑ → FP↓ (낮은 신뢰도 박스 제거)"
echo "• 최적 조합으로 H-Mean 96.3-96.5% 달성 예상"
echo ""
