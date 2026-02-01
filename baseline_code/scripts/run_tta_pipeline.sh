#!/bin/bash

BASE_DIR="/data/ephemeral/home/baseline_code"
cd $BASE_DIR

echo "========================================"
echo "TTA Prediction (Original + HFlip)"
echo "========================================"

# 1. 원본 예측 (이미 완료)
echo "Step 1: Using existing original predictions..."
ORIG_JSON=$(ls -t outputs/resnet50_fold0_aggressive/submissions/*.json | head -1)
echo "Original: $ORIG_JSON"

# 2. HFlip 예측
echo ""
echo "Step 2: Predicting with horizontal flip..."
python runners/predict.py --config-name=predict_resnet50_hflip

FLIP_JSON=$(ls -t outputs/resnet50_hflip_predict/submissions/*.json | head -1)
echo "Flipped: $FLIP_JSON"

# 3. 병합
echo ""
echo "Step 3: Merging predictions..."
python scripts/merge_tta_predictions.py \
    "$ORIG_JSON" \
    "$FLIP_JSON" \
    outputs/tta_merged.json \
    0.3

# 4. CSV 변환
echo ""
echo "Step 4: Converting to CSV..."
python ocr/utils/convert_submission.py \
    -J outputs/tta_merged.json \
    -O outputs/submission_resnet50_tta.csv <<< "yes"

echo ""
echo "========================================"
echo "TTA Complete!"
echo "File: outputs/submission_resnet50_tta.csv"
echo "========================================"
