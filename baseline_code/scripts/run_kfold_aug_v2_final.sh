#!/bin/bash

# K-Fold Training Script for Augmentation v2 (960px + Heavy Aug)
export CUDA_VISIBLE_DEVICES=0

# Load environment variables
set -a
source /data/ephemeral/home/baseline_code/.env
set +a

BASE_DIR="/data/ephemeral/home/baseline_code"
LOG_DIR="$BASE_DIR/logs"
mkdir -p $LOG_DIR

cd $BASE_DIR

echo "Starting K-Fold Training (Augmentation v2) at $(date)"

for FOLD in 0 1 2 3 4; do
    echo "======================================"
    echo "Training Fold $FOLD"
    echo "======================================"
    
    python runners/train.py \
        preset=augmented_v2 \
        ++datasets.train_dataset.annotation_path="$BASE_DIR/kfold_results_v2/fold_${FOLD}/train.json" \
        ++datasets.val_dataset.annotation_path="$BASE_DIR/kfold_results_v2/fold_${FOLD}/val.json" \
        ++trainer.max_epochs=24 \
        exp_name="aug_v2_fold${FOLD}" \
        wandb=True \
        > "$LOG_DIR/fold${FOLD}_aug_v2_final.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Fold $FOLD training completed successfully"
    else
        echo "Fold $FOLD training failed!"
        exit 1
    fi
done

echo "All K-Fold training completed at $(date)"
