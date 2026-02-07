#!/bin/bash

echo "======================================================================"
echo "K-FOLD ENSEMBLE PREDICTION - Using Test Mode"
echo "======================================================================"
echo ""
echo "This script will:"
echo "1. Run test inference on all 5 folds"
echo "2. Each fold saves predictions in outputs/*/submissions/"
echo "3. Collect all predictions for ensemble"
echo ""

BASE_DIR="/data/ephemeral/home/baseline_code"
cd "$BASE_DIR"

# Fold checkpoints
declare -A CHECKPOINTS
CHECKPOINTS[0]="outputs/hrnet_w44_1280_optimal_fold0/checkpoints/epoch=4-step=6545.ckpt"
CHECKPOINTS[1]="outputs/hrnet_w44_1280_optimal_fold1/checkpoints/epoch=3-step=5236.ckpt"
CHECKPOINTS[2]="outputs/hrnet_w44_1280_optimal_fold2/checkpoints/epoch=18-step=24871.ckpt"
CHECKPOINTS[3]="outputs/hrnet_w44_1280_optimal_fold3/checkpoints/epoch=4-step=6545.ckpt"
CHECKPOINTS[4]="outputs/hrnet_w44_1280_optimal_fold4/checkpoints/epoch=17-step=23544.ckpt"

SUCCESS_COUNT=0

# Run test for each fold
for i in {0..4}; do
    echo ""
    echo "======================================================================"
    echo "Fold $i: Running test inference..."
    echo "Checkpoint: ${CHECKPOINTS[$i]}"
    echo "======================================================================"
    echo ""
    
    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINTS[$i]}" ]; then
        echo "❌ Checkpoint not found: ${CHECKPOINTS[$i]}"
        continue
    fi
    
    # Run test (will save predictions automatically)
    python runners/test.py \
        preset=hrnet_w44_1280 \
        checkpoint_path="${CHECKPOINTS[$i]}" \
        wandb=false \
        2>&1 | grep -E "Testing|test/|Predictions saved|ERROR" || true
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Fold $i test complete"
        ((SUCCESS_COUNT++))
    else
        echo "❌ Fold $i test failed"
    fi
    
    echo ""
    sleep 2
done

echo "======================================================================"
echo "Test Summary: $SUCCESS_COUNT/5 folds completed"
echo "======================================================================"
echo ""

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo "❌ All tests failed"
    exit 1
fi

echo "✓ Test predictions generated!"
echo ""
echo "Next steps:"
echo "  1. Check outputs/hrnet_w44_1280_optimal_fold*/submissions/"
echo "  2. Collect and ensemble predictions manually"
echo "  3. Or use existing ensemble script"
echo ""
