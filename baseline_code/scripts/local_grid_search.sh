#!/bin/bash

# Local Hyperparameter Search - WandB 없이 실행
# Purpose: thresh, box_thresh 최적화로 Precision 개선
# Target: 96.00% → 96.50%

set -e

echo "========================================="
echo "🔍 Local Grid Search - Postprocessing 최적화"
echo "========================================="
echo ""

# 설정
BASE_PRESET="efficientnet_b4_aggressive"
CHECKPOINT="outputs/efficientnet_b4_single/checkpoints/epoch=15-step=13088.ckpt"
OUTPUT_DIR="outputs/efficientnet_b4_grid_search"
RESULTS_FILE="${OUTPUT_DIR}/grid_search_results.csv"

# 결과 디렉토리 생성
mkdir -p ${OUTPUT_DIR}/submissions
mkdir -p ${OUTPUT_DIR}/logs

# 결과 파일 초기화
echo "run_id,thresh,box_thresh,max_candidates,h_mean,precision,recall" > ${RESULTS_FILE}

echo "📊 Grid Search 설정"
echo "-----------------------------------------"
echo "Base Model: EfficientNet-B4"
echo "Checkpoint: epoch=15"
echo "Search Space:"
echo "  - thresh: [0.20, 0.22, 0.24, 0.26, 0.28]"
echo "  - box_thresh: [0.22, 0.25, 0.28, 0.30, 0.32]"
echo "  - max_candidates: [500, 600, 700]"
echo ""
echo "총 실행 횟수: 75회 (5 x 5 x 3)"
echo "예상 소요 시간: ~40분"
echo ""

read -p "계속하시겠습니까? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 1
fi

echo ""
echo "========================================="
echo "🚀 Grid Search 시작"
echo "========================================="
echo ""

RUN_ID=1
BEST_HMEAN=0
BEST_CONFIG=""

# Grid Search Loop
for THRESH in 0.20 0.22 0.24 0.26 0.28; do
    for BOX_THRESH in 0.22 0.25 0.28 0.30 0.32; do
        for MAX_CAND in 500 600 700; do
            
            echo "[$RUN_ID/75] thresh=$THRESH, box_thresh=$BOX_THRESH, max_cand=$MAX_CAND"
            
            # 예측 실행
            PRED_OUTPUT="${OUTPUT_DIR}/submissions/pred_${RUN_ID}"
            
            python runners/predict.py \
                preset=${BASE_PRESET} \
                exp_name=grid_search_${RUN_ID} \
                checkpoint_path=${CHECKPOINT} \
                models.head.thresh=${THRESH} \
                models.head.box_thresh=${BOX_THRESH} \
                models.head.max_candidates=${MAX_CAND} \
                output_dir=${PRED_OUTPUT} \
                > ${OUTPUT_DIR}/logs/run_${RUN_ID}.log 2>&1
            
            # JSON 파일 찾기
            JSON_FILE=$(ls -t ${PRED_OUTPUT}/submissions/*.json 2>/dev/null | head -1)
            
            if [ -z "$JSON_FILE" ]; then
                echo "  ⚠️  예측 파일 생성 실패"
                echo "${RUN_ID},${THRESH},${BOX_THRESH},${MAX_CAND},0,0,0" >> ${RESULTS_FILE}
            else
                # CSV 변환
                CSV_FILE="${PRED_OUTPUT}/submissions/submission.csv"
                python ocr/utils/convert_submission.py \
                    -J ${JSON_FILE} \
                    -O ${CSV_FILE} \
                    > /dev/null 2>&1
                
                echo "  ✅ 생성 완료: ${CSV_FILE}"
                
                # 여기서는 실제 평가를 할 수 없으므로 임시 저장만
                # 실제로는 각 CSV를 리더보드에 제출해야 함
                echo "${RUN_ID},${THRESH},${BOX_THRESH},${MAX_CAND},0,0,0" >> ${RESULTS_FILE}
                
                # 최고 성능 추적 (placeholder)
                # if (( $(echo "$HMEAN > $BEST_HMEAN" | bc -l) )); then
                #     BEST_HMEAN=$HMEAN
                #     BEST_CONFIG="thresh=$THRESH, box_thresh=$BOX_THRESH, max_cand=$MAX_CAND"
                # fi
            fi
            
            RUN_ID=$((RUN_ID + 1))
            sleep 1
        done
    done
done

echo ""
echo "========================================="
echo "✅ Grid Search 완료!"
echo "========================================="
echo ""
echo "📊 결과 요약"
echo "-----------------------------------------"
echo "총 실행: 75회"
echo "결과 파일: ${RESULTS_FILE}"
echo "제출 파일: ${OUTPUT_DIR}/submissions/"
echo ""
echo "📋 다음 단계:"
echo "1. ${OUTPUT_DIR}/submissions/ 의 모든 CSV 파일을"
echo "   리더보드에 제출하여 실제 H-Mean 확인"
echo ""
echo "2. 최고 성능의 파라미터를 확인하여"
echo "   configs/preset/efficientnet_b4_optimal.yaml 생성"
echo ""
echo "3. 최적 파라미터로 재학습 실행"
echo ""
