#!/bin/bash

echo "========================================"
echo "Starting ResNet50 Fold 2 Training"
echo "Time: $(date)"
echo "========================================"

cd /data/ephemeral/home/baseline_code

# Set environment
export PYTHONPATH=/data/ephemeral/home/baseline_code:$PYTHONPATH
source /data/ephemeral/home/.env

# Train Fold 2 with aggressive settings
python runners/train.py \
    preset=augmented_resnet50_aggressive_fold2 \
    exp_name=resnet50_fold2 \
    trainer.max_epochs=22

echo "========================================"
echo "ResNet50 Fold 2 training completed"
echo "Time: $(date)"
echo "========================================"
