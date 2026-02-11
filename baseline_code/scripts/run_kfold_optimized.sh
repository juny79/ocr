#!/bin/bash
# K-Fold Cross-Validation Training Script (Optimized Parameters)
# 
# 최적 파라미터 조합:
# - Sweep 최적: LR=0.001336, WD=0.000357, T_max=12
# - Loss 최적: negative_ratio=2.824, prob_weight=3.591, thresh_weight=8.029
# - Grid Search: thresh=0.220, box_thresh=0.400
# 
# 목표: H-Mean 0.9855+ 달성 (통합 데이터셋 K-Fold)

set -e

export WANDB_API_KEY="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"
export WANDB_PROJECT="ocr-receipt-detection"
export WANDB_ENTITY="fc_bootcamp"

cd /data/ephemeral/home/baseline_code

echo "======================================================================="
echo "K-FOLD TRAINING - OPTIMIZED PARAMETERS"
echo "======================================================================="
echo "Sweep 최적 파라미터:"
echo "  - Learning Rate: 0.001336"
echo "  - Weight Decay: 0.000357"
echo "  - T_max: 12 (CosineAnnealingLR)"
echo ""
echo "Loss 최적화 파라미터:"
echo "  - negative_ratio: 2.824"
echo "  - prob_map_loss_weight: 3.591"
echo "  - thresh_map_loss_weight: 8.029"
echo "  - binary_map_loss_weight: 0.692"
echo ""
echo "Grid Search 후처리 파라미터:"
echo "  - thresh: 0.220"
echo "  - box_thresh: 0.400"
echo ""
echo "학습 설정:"
echo "  - Max Epochs: 13"
echo "  - Resolution: 1024x1024"
echo "  - Model: HRNet-W44"
echo "  - Optimizer: AdamW"
echo "======================================================================="
echo ""

# Training loop
for fold in 0 1 2 3 4; do
    echo ""
    echo "======================================================================="
    echo "Training Fold $((fold))/4 (Fold ID: ${fold})"
    echo "======================================================================="
    
    # Set fold-specific annotation paths
    TRAIN_ANN="/data/ephemeral/home/data/datasets/jsons/kfold/fold${fold}_train.json"
    VAL_ANN="/data/ephemeral/home/data/datasets/jsons/kfold/fold${fold}_val.json"
    
    # Check if annotation files exist
    if [ ! -f "$TRAIN_ANN" ]; then
        echo "Error: Training annotation file not found: $TRAIN_ANN"
        echo "Please run K-Fold split creation first."
        exit 1
    fi
    
    if [ ! -f "$VAL_ANN" ]; then
        echo "Error: Validation annotation file not found: $VAL_ANN"
        echo "Please run K-Fold split creation first."
        exit 1
    fi
    
    # Run training with optimized parameters
    python runners/train.py \
        preset=hrnet_w44_1024 \
        wandb=True \
        wandb_config.tags='["kfold","optimized","sweep_best","grid_search"]' \
        exp_name="hrnet_w44_kfold_fold${fold}_optimized" \
        datasets.train_dataset.annotation_path="$TRAIN_ANN" \
        datasets.val_dataset.annotation_path="$VAL_ANN" \
        datasets.val_dataset.image_path="/data/ephemeral/home/data/datasets/images/train" \
        models.optimizer.lr=0.001336 \
        models.optimizer.weight_decay=0.000357 \
        models.scheduler.T_max=12 \
        models.loss.negative_ratio=2.824132345320219 \
        models.loss.prob_map_loss_weight=3.591196851512631 \
        models.loss.thresh_map_loss_weight=8.028627860143937 \
        models.loss.binary_map_loss_weight=0.6919312670387725 \
        models.head.postprocess.thresh=0.220 \
        models.head.postprocess.box_thresh=0.400 \
        trainer.max_epochs=13 \
        checkpoint_dir="checkpoints/kfold_optimized/fold_${fold}" \
        2>&1 | tee "logs/kfold_optimized_fold${fold}.log"
    
    echo "✓ Fold ${fold} training completed"
    echo "  Log: logs/kfold_optimized_fold${fold}.log"
    echo "  Checkpoint: checkpoints/kfold_optimized/fold_${fold}/"
    echo ""
done

echo ""
echo "======================================================================="
echo "✓ K-FOLD TRAINING COMPLETE!"
echo "======================================================================="
echo ""
echo "결과 위치:"
echo "  - Checkpoints: checkpoints/kfold_optimized/fold_*/"
echo "  - Logs: logs/kfold_optimized_fold*.log"
echo ""
echo "다음 단계:"
echo "  1. 각 fold 모델로 예측 수행"
echo "  2. 앙상블 (5-Fold 평균)"
echo "  3. 후처리 최적화 (thresh=0.220, box_thresh=0.400)"
echo "======================================================================="
