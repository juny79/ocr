#!/bin/bash
# Auto-train K-Fold with Augmented Strategy v2 (960px)

# Export WandB API Key
export WANDB_API_KEY="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"

cd /data/ephemeral/home/baseline_code

echo "========================================="
echo " Augmented K-Fold Training Start (v2)"
echo " Image Size: 960px"
echo " Strategy: Heavy Augmentation"
echo "========================================="

# Train all 5 folds
for fold in 0 1 2 3 4; do
    echo ""
    echo "Starting Fold ${fold}..."
    LOG_FILE="logs/fold${fold}_aug_v2_train.log"
    
    # Run training in background with nohup
    # Override dataset paths for K-Fold
    # Increase max_epochs to 25 as requested
    # Fix: Correct JSON paths for K-Fold
    nohup python runners/train.py \
        preset=augmented_v2 \
        +trainer.enable_progress_bar=False \
        datasets.train_dataset.annotation_path="/data/ephemeral/home/baseline_code/kfold_results/fold_${fold}/train.json" \
        datasets.val_dataset.annotation_path="/data/ephemeral/home/baseline_code/kfold_results/fold_${fold}/val.json" \
        datasets.val_dataset.image_path="/data/ephemeral/home/data/datasets/images/train" \
        checkpoint_dir="checkpoints/kfold_aug_v2/fold_${fold}" \
        trainer.max_epochs=25 \
        models.encoder.model_name="resnet34" \
        exp_name="aug_v2_fold_${fold}" \
        wandb=True \
        > ${LOG_FILE} 2>&1 &
        
    PID=$!
    echo "✓ Fold ${fold} started (PID: ${PID})"
    echo "  Log: ${LOG_FILE}"
    
    # Wait for the process to finish before starting next fold?
    # Usually sequential is safer for GPU memory, unless multiple GPUs.
    # Assuming single GPU, we must wait.
    wait $PID
    
    echo "✓ Fold ${fold} completed!"
done

echo ""
echo "========================================="
echo "All augmented folds completed!"
echo "========================================="
