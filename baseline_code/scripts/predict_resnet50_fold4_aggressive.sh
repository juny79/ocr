#!/bin/bash

echo "========================================"
echo "Starting ResNet50 Fold 4 Prediction (Aggressive)"
echo "Time: $(date)"
echo "========================================"

cd /data/ephemeral/home/baseline_code

# Set environment
export PYTHONPATH=/data/ephemeral/home/baseline_code:$PYTHONPATH
source /data/ephemeral/home/.env

# Find best checkpoint (latest epoch)
CHECKPOINT=$(ls -t outputs/resnet50_fold4/checkpoints/*.ckpt | head -1)
# Escape = signs in checkpoint path for Hydra
CHECKPOINT_ESCAPED=$(echo $CHECKPOINT | sed 's/=/\\=/g')
echo "Using checkpoint: $CHECKPOINT"

# Run prediction with aggressive settings
python runners/predict.py \
    preset=augmented_resnet50_aggressive \
    checkpoint_path=$CHECKPOINT_ESCAPED \
    exp_name=resnet50_fold4_aggressive_predict

echo "========================================"
echo "ResNet50 Fold 4 prediction completed"
echo "Time: $(date)"
echo "========================================"

# Show generated JSON
JSON_FILE=$(ls -t outputs/resnet50_fold4_aggressive_predict/submissions/*.json 2>/dev/null | head -1)
if [ -n "$JSON_FILE" ]; then
    echo "Generated JSON: $JSON_FILE"
else
    echo "Warning: JSON file not found"
fi
