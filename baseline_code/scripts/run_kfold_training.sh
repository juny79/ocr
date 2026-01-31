#!/bin/bash
# K-Fold Cross-Validation Training Script
# Runs 5 separate training sessions with different fold splits

set -e

export WANDB_API_KEY="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"

cd /data/ephemeral/home/baseline_code

echo "======================================================================="
echo "K-FOLD CROSS-VALIDATION TRAINING (5 Folds)"
echo "Scheduler: CosineAnnealingLR (T_max=20)"
echo "Max Epochs: 20 | Early Stopping: patience=5"
echo "======================================================================="
echo ""

# Train each fold
for fold in 0 1 2 3 4; do
    echo ""
    echo "======================================================================="
    echo "Training Fold $((fold + 1))/5"
    echo "======================================================================="
    
    # Set fold-specific annotation paths
    TRAIN_ANN="/data/ephemeral/home/data/datasets/jsons/kfold/fold${fold}_train.json"
    VAL_ANN="/data/ephemeral/home/data/datasets/jsons/kfold/fold${fold}_val.json"
    
    # Run training with fold-specific data
    python runners/train.py \
        preset=example \
        wandb=True \
        datasets.train_dataset.annotation_path="$TRAIN_ANN" \
        datasets.val_dataset.annotation_path="$VAL_ANN" \
        datasets.val_dataset.image_path="/data/ephemeral/home/data/datasets/images/train" \
        checkpoint_dir="checkpoints/kfold/fold_${fold}" \
        2>&1 | tee "logs/kfold_fold${fold}_training.log"
    
    echo "✓ Fold $((fold + 1))/5 completed"
    echo ""
done

echo ""
echo "======================================================================="
echo "✓ K-FOLD CROSS-VALIDATION TRAINING COMPLETE!"
echo "Checkpoints saved in: checkpoints/kfold/fold_*/"
echo "Logs saved in: logs/kfold_fold*_training.log"
echo "======================================================================="
