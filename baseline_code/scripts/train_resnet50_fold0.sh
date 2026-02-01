#!/bin/bash

# ResNet50 Single Training Script
export CUDA_VISIBLE_DEVICES=0

# Load environment variables
set -a
source /data/ephemeral/home/baseline_code/.env
set +a

BASE_DIR="/data/ephemeral/home/baseline_code"
LOG_DIR="$BASE_DIR/logs"
mkdir -p $LOG_DIR

cd $BASE_DIR

echo "========================================"
echo "Starting ResNet50 Training"
echo "Time: $(date)"
echo "========================================"

python runners/train.py \
    preset=augmented_resnet50 \
    ++datasets.train_dataset.annotation_path="$BASE_DIR/kfold_results_v2/fold_0/train.json" \
    ++datasets.val_dataset.annotation_path="$BASE_DIR/kfold_results_v2/fold_0/val.json" \
    ++trainer.max_epochs=24 \
    exp_name="resnet50_fold0" \
    wandb=True

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "ResNet50 training completed successfully"
    echo "Time: $(date)"
    echo "========================================"
else
    echo "Training failed!"
    exit 1
fi
