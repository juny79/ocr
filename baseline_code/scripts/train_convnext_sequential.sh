#!/bin/bash
# ConvNeXt 모델 순차 학습 스크립트

set -e

BASE_DIR="/data/ephemeral/home/baseline_code"
cd $BASE_DIR

echo "================================================================================"
echo "🔬 ConvNeXt Model Comparison - Sequential Training"
echo "================================================================================"
echo ""

# Hyperparameters (same as EfficientNet-B3)
HPARAMS="models.optimizer.lr=0.00045 \
models.optimizer.weight_decay=0.000085 \
models.scheduler.T_max=20 \
models.scheduler.eta_min=0.000008 \
trainer.max_epochs=20 \
wandb=true"

# Train ConvNeXt-Tiny
echo "📌 Step 1/2: Training ConvNeXt-Tiny"
echo "────────────────────────────────────────────────────────────────────────────────"
python runners/train_convnext_tiny.py preset=convnext_tiny_hybrid $HPARAMS 2>&1 | tee convnext_tiny_full.log
TINY_STATUS=$?

if [ $TINY_STATUS -eq 0 ]; then
    echo "✅ ConvNeXt-Tiny training completed successfully"
    TINY_RESULT=$(grep "best_model_score" convnext_tiny_full.log | tail -1 || echo "N/A")
else
    echo "❌ ConvNeXt-Tiny training failed with status: $TINY_STATUS"
    TINY_RESULT="FAILED"
fi

echo ""
echo "================================================================================"
echo ""

# Train ConvNeXt-Small
echo "📌 Step 2/2: Training ConvNeXt-Small"
echo "────────────────────────────────────────────────────────────────────────────────"
python runners/train_convnext_small.py preset=convnext_small_hybrid $HPARAMS 2>&1 | tee convnext_small_full.log
SMALL_STATUS=$?

if [ $SMALL_STATUS -eq 0 ]; then
    echo "✅ ConvNeXt-Small training completed successfully"
    SMALL_RESULT=$(grep "best_model_score" convnext_small_full.log | tail -1 || echo "N/A")
else
    echo "❌ ConvNeXt-Small training failed with status: $SMALL_STATUS"
    SMALL_RESULT="FAILED"
fi

echo ""
echo "================================================================================"
echo "📊 TRAINING SUMMARY"
echo "================================================================================"
echo ""

# Extract best scores
TINY_SCORE="N/A"
SMALL_SCORE="N/A"

if [ -f "outputs/convnext_tiny_hybrid_progressive_fold0/checkpoints/fold_0/best-epoch*.ckpt" ]; then
    TINY_SCORE=$(ls outputs/convnext_tiny_hybrid_progressive_fold0/checkpoints/fold_0/best-epoch*.ckpt | grep -oP 'hmean=\K[0-9.]+' | sort -rn | head -1 || echo "N/A")
fi

if [ -f "outputs/convnext_small_hybrid_progressive_fold0/checkpoints/fold_0/best-epoch*.ckpt" ]; then
    SMALL_SCORE=$(ls outputs/convnext_small_hybrid_progressive_fold0/checkpoints/fold_0/best-epoch*.ckpt | grep -oP 'hmean=\K[0-9.]+' | sort -rn | head -1 || echo "N/A")
fi

echo "┌─────────────────┬───────────────┬──────────┐"
echo "│ Model           │ Val H-Mean    │ Status   │"
echo "├─────────────────┼───────────────┼──────────┤"
echo "│ EfficientNet-B3 │ 0.9658        │ ✓        │"
echo "│ ConvNeXt-Tiny   │ ${TINY_SCORE:-N/A}        │ $([ $TINY_STATUS -eq 0 ] && echo '✓' || echo '✗')        │"
echo "│ ConvNeXt-Small  │ ${SMALL_SCORE:-N/A}        │ $([ $SMALL_STATUS -eq 0 ] && echo '✓' || echo '✗')        │"
echo "└─────────────────┴───────────────┴──────────┘"
echo ""

echo "📂 Checkpoint Locations:"
echo "  • ConvNeXt-Tiny:  outputs/convnext_tiny_hybrid_progressive_fold0/checkpoints/fold_0/"
echo "  • ConvNeXt-Small: outputs/convnext_small_hybrid_progressive_fold0/checkpoints/fold_0/"
echo ""

echo "================================================================================"
echo "✅ All training completed at $(date)"
echo "================================================================================"
