#!/bin/bash
# Augmented V2 K-Fold Training (960px, Heavy Augmentation)
# Uses merged image folder (images/all) and kfold_results_v2

export WANDB_API_KEY="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"
cd /data/ephemeral/home/baseline_code

echo "========================================="
echo " Augmented V2 K-Fold Training"
echo " Resolution: 960px | Batch: 8"
echo " Epochs: 25 | Backbone: ResNet34"
echo "========================================="

for fold in 0 1 2 3 4; do
    echo ""
    echo "=== Starting Fold ${fold} ==="
    
    python runners/train.py \
        preset=augmented_v2 \
        datasets.train_dataset.annotation_path="/data/ephemeral/home/baseline_code/kfold_results_v2/fold_${fold}/train.json" \
        datasets.val_dataset.annotation_path="/data/ephemeral/home/baseline_code/kfold_results_v2/fold_${fold}/val.json" \
        checkpoint_dir="checkpoints/kfold_aug_v2/fold_${fold}" \
        trainer.max_epochs=25 \
        models.encoder.model_name="resnet34" \
        exp_name="aug_v2_fold_${fold}" \
        wandb=True
    
    echo "=== Fold ${fold} Completed ==="
done

echo ""
echo "========================================="
echo " All 5 Folds Completed!"
echo "========================================="
