#!/bin/bash
# Single Fold Training Script
# Usage: bash train_single_fold.sh <fold_number>

if [ -z "$1" ]; then
    echo "Usage: bash train_single_fold.sh <fold_number>"
    echo "Example: bash train_single_fold.sh 1"
    exit 1
fi

FOLD=$1

export WANDB_API_KEY="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"

cd /data/ephemeral/home/baseline_code

echo "======================================================================="
echo "Training Fold ${FOLD}"
echo "======================================================================="

TRAIN_ANN="/data/ephemeral/home/data/datasets/jsons/kfold/fold${FOLD}_train.json"
VAL_ANN="/data/ephemeral/home/data/datasets/jsons/kfold/fold${FOLD}_val.json"

python runners/train.py \
    preset=example \
    wandb=True \
    datasets.train_dataset.annotation_path="$TRAIN_ANN" \
    datasets.val_dataset.annotation_path="$VAL_ANN" \
    datasets.val_dataset.image_path="/data/ephemeral/home/data/datasets/images/train" \
    checkpoint_dir="checkpoints/kfold/fold_${FOLD}"

echo "✓ Fold ${FOLD} training completed"
