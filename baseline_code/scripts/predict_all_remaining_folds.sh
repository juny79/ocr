#!/bin/bash

echo "========================================"
echo "Fold 2, 3, 4 Prediction Pipeline"
echo "Starting: $(date)"
echo "========================================"

BASE_DIR="/data/ephemeral/home/baseline_code"
cd $BASE_DIR

# Predict Fold 2
echo ""
echo "=== Predicting Fold 2 ==="
bash scripts/predict_resnet50_fold2_aggressive.sh
if [ $? -ne 0 ]; then
    echo "❌ Fold 2 prediction failed!"
    exit 1
fi
echo "✅ Fold 2 completed"

# Predict Fold 3
echo ""
echo "=== Predicting Fold 3 ==="
bash scripts/predict_resnet50_fold3_aggressive.sh
if [ $? -ne 0 ]; then
    echo "❌ Fold 3 prediction failed!"
    exit 1
fi
echo "✅ Fold 3 completed"

# Predict Fold 4
echo ""
echo "=== Predicting Fold 4 ==="
bash scripts/predict_resnet50_fold4_aggressive.sh
if [ $? -ne 0 ]; then
    echo "❌ Fold 4 prediction failed!"
    exit 1
fi
echo "✅ Fold 4 completed"

# Summary
echo ""
echo "========================================"
echo "All Predictions Complete!"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Generated JSON files:"
for fold in 2 3 4; do
    JSON=$(ls -t outputs/resnet50_fold${fold}_aggressive_predict/submissions/*.json 2>/dev/null | head -1)
    if [ -n "$JSON" ]; then
        echo "  Fold $fold: $JSON"
    else
        echo "  Fold $fold: ❌ Not found"
    fi
done

echo ""
echo "Next step: Run 5-Fold ensemble"
echo "  python scripts/ensemble_5fold.py \\"
echo "    --fold0 outputs/resnet50_fold0_aggressive/submissions/XXXXX.json \\"
echo "    --fold1 outputs/resnet50_fold1_aggressive_predict/submissions/XXXXX.json \\"
echo "    --fold2 outputs/resnet50_fold2_aggressive_predict/submissions/XXXXX.json \\"
echo "    --fold3 outputs/resnet50_fold3_aggressive_predict/submissions/XXXXX.json \\"
echo "    --fold4 outputs/resnet50_fold4_aggressive_predict/submissions/XXXXX.json \\"
echo "    --output outputs/ensemble_5fold_voting3.json \\"
echo "    --voting 3"
