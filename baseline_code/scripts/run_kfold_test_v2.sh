#!/bin/bash

echo "======================================================================"
echo "K-FOLD ENSEMBLE PREDICTION - Creating Config Files"
echo "======================================================================"
echo ""

BASE_DIR="/data/ephemeral/home/baseline_code"
cd "$BASE_DIR"

# Fold checkpoints
declare -a CHECKPOINTS=(
    "outputs/hrnet_w44_1280_optimal_fold0/checkpoints/epoch=4-step=6545.ckpt"
    "outputs/hrnet_w44_1280_optimal_fold1/checkpoints/epoch=3-step=5236.ckpt"
    "outputs/hrnet_w44_1280_optimal_fold2/checkpoints/epoch=18-step=24871.ckpt"
    "outputs/hrnet_w44_1280_optimal_fold3/checkpoints/epoch=4-step=6545.ckpt"
    "outputs/hrnet_w44_1280_optimal_fold4/checkpoints/epoch=17-step=23544.ckpt"
)

SUCCESS_COUNT=0

# Create temp configs directory
TEMP_CONFIGS="$BASE_DIR/temp_kfold_configs"
mkdir -p "$TEMP_CONFIGS"

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
    
    # Create temporary config file
    CONFIG_FILE="$TEMP_CONFIGS/fold_${i}_test.yaml"
    cat > "$CONFIG_FILE" << EOF
defaults:
  - _self_
  - preset: hrnet_w44_1280
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

seed: 42
exp_name: fold_${i}_test
wandb: false

checkpoint_path: "${CHECKPOINTS[$i]}"
EOF
    
    echo "Created config: $CONFIG_FILE"
    
    # Run test using config file
    python runners/test.py \
        --config-path "$TEMP_CONFIGS" \
        --config-name "fold_${i}_test" \
        2>&1 | tail -50
    
    RESULT=${PIPESTATUS[0]}
    
    if [ $RESULT -eq 0 ]; then
        echo "✓ Fold $i test complete"
        ((SUCCESS_COUNT++))
    else
        echo "❌ Fold $i test failed (exit code: $RESULT)"
    fi
    
    echo ""
    sleep 2
done

echo "======================================================================"
echo "Test Summary: $SUCCESS_COUNT/5 folds completed"
echo "======================================================================"
echo ""

# Cleanup temp configs
#rm -rf "$TEMP_CONFIGS"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "✓ $SUCCESS_COUNT fold predictions generated!"
    echo ""
    echo "Check outputs/hrnet_w44_1280_optimal_fold*/submissions/ for results"
    echo ""
else
    echo "❌ All tests failed"
    exit 1
fi
