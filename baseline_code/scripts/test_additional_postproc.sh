#!/bin/bash

# Additional Postprocessing Tests
# Testing thresh 0.29, 0.30 combinations

set -e

CHECKPOINT="checkpoints/epoch=15-step=13088.ckpt"
OUTPUT_DIR="outputs/efficientnet_b4_postproc_final"

echo "=================================================="
echo "Postprocessing 추가 테스트"
echo "=================================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "테스트 조합: 4가지"
echo ""

# Test combinations
combinations=(
    "0.29 0.25"
    "0.29 0.24"
    "0.30 0.25"
    "0.30 0.26"
)

for combo in "${combinations[@]}"; do
    read thresh box_thresh <<< "$combo"
    
    echo "=================================================="
    echo "Testing: thresh=${thresh}, box_thresh=${box_thresh}"
    echo "=================================================="
    
    python runners/predict.py \
        preset=efficientnet_b4_lr_optimized \
        model_path=${CHECKPOINT} \
        exp_name=efficientnet_b4_postproc_final \
        output_dir=${OUTPUT_DIR} \
        submission_name=submission_t${thresh}_b${box_thresh}.csv \
        'models.head.thresh='${thresh} \
        'models.head.box_thresh='${box_thresh} \
        models.head.max_candidates=600
    
    echo ""
    echo "✅ Completed: thresh=${thresh}, box_thresh=${box_thresh}"
    echo "   Output: ${OUTPUT_DIR}/submissions/submission_t${thresh}_b${box_thresh}.csv"
    echo ""
done

echo "=================================================="
echo "✅ 모든 테스트 완료!"
echo "=================================================="
echo ""
echo "생성된 파일:"
ls -lh ${OUTPUT_DIR}/submissions/submission_t0.*.csv
echo ""
echo "다음 단계: 리더보드에 제출하여 최적값 확인"
