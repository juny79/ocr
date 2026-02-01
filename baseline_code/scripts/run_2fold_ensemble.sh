#!/bin/bash

BASE_DIR="/data/ephemeral/home/baseline_code"
cd $BASE_DIR

echo "========================================"
echo "2-Fold Ensemble Pipeline"
echo "========================================"

# 1. Fold 0 예측 (aggressive 설정, 이미 완료)
echo "Step 1: Using existing Fold 0 predictions (aggressive)..."
FOLD0_JSON="outputs/resnet50_fold0_aggressive/submissions/20260201_052309.json"
echo "Fold 0: $FOLD0_JSON"

# 2. Fold 1 예측 (aggressive 설정)
echo ""
echo "Step 2: Predicting with Fold 1 model (aggressive)..."

# Fold 1 최적 체크포인트 찾기
FOLD1_CKPT=$(ls -t outputs/resnet50_fold1/checkpoints/*.ckpt | grep -E "epoch=[0-9]+-step=" | head -1)

if [ -z "$FOLD1_CKPT" ]; then
    echo "Error: Fold 1 checkpoint not found!"
    echo "Please wait for Fold 1 training to complete."
    exit 1
fi

echo "Using checkpoint: $FOLD1_CKPT"

# Fold 1 예측 실행
python runners/predict.py \
    preset=augmented_resnet50_aggressive \
    ++checkpoint_path="$FOLD1_CKPT" \
    exp_name="resnet50_fold1_aggressive_predict"

FOLD1_JSON=$(ls -t outputs/resnet50_fold1_aggressive_predict/submissions/*.json | head -1)
echo "Fold 1: $FOLD1_JSON"

# 3. 앙상블 (Voting >= 1)
echo ""
echo "Step 3: Ensemble (2-Fold, Voting >= 1)..."
python scripts/ensemble_2fold.py \
    --fold0 "$FOLD0_JSON" \
    --fold1 "$FOLD1_JSON" \
    --output outputs/ensemble_2fold_voting1.json \
    --voting 1 \
    --iou 0.5

# 4. CSV 변환
echo ""
echo "Step 4: Converting to CSV..."
python ocr/utils/convert_submission.py \
    -J outputs/ensemble_2fold_voting1.json \
    -O outputs/submission_resnet50_2fold_ensemble.csv <<< "yes"

echo ""
echo "========================================"
echo "Ensemble Complete!"
echo "File: outputs/submission_resnet50_2fold_ensemble.csv"
echo "========================================"
