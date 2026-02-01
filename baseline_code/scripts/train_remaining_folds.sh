#!/bin/bash

echo "========================================"
echo "5-Fold Training Pipeline (Fold 2, 3, 4)"
echo "Starting: $(date)"
echo "========================================"

BASE_DIR="/data/ephemeral/home/baseline_code"
cd $BASE_DIR

# Check if Fold 0 and 1 are completed
echo "Checking existing folds..."
if [ ! -d "outputs/resnet50_fold0/checkpoints" ]; then
    echo "❌ Error: Fold 0 checkpoints not found!"
    exit 1
fi

if [ ! -d "outputs/resnet50_fold1/checkpoints" ]; then
    echo "❌ Error: Fold 1 checkpoints not found!"
    exit 1
fi

echo "✅ Fold 0: $(ls outputs/resnet50_fold0/checkpoints/*.ckpt | wc -l) checkpoints"
echo "✅ Fold 1: $(ls outputs/resnet50_fold1/checkpoints/*.ckpt | wc -l) checkpoints"
echo ""

# Train Fold 2
echo "========================================"
echo "Training Fold 2/4"
echo "========================================"
bash scripts/train_resnet50_fold2.sh
if [ $? -ne 0 ]; then
    echo "❌ Fold 2 training failed!"
    exit 1
fi
echo "✅ Fold 2 completed"
echo ""

# Train Fold 3
echo "========================================"
echo "Training Fold 3/4"
echo "========================================"
bash scripts/train_resnet50_fold3.sh
if [ $? -ne 0 ]; then
    echo "❌ Fold 3 training failed!"
    exit 1
fi
echo "✅ Fold 3 completed"
echo ""

# Train Fold 4
echo "========================================"
echo "Training Fold 4/4"
echo "========================================"
bash scripts/train_resnet50_fold4.sh
if [ $? -ne 0 ]; then
    echo "❌ Fold 4 training failed!"
    exit 1
fi
echo "✅ Fold 4 completed"
echo ""

# Summary
echo "========================================"
echo "5-Fold Training Complete!"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Checkpoint Summary:"
for fold in 0 1 2 3 4; do
    ckpt_count=$(ls outputs/resnet50_fold${fold}/checkpoints/*.ckpt 2>/dev/null | wc -l)
    echo "  Fold ${fold}: ${ckpt_count} checkpoints"
done

echo ""
echo "Next steps:"
echo "  1. Run predictions for Fold 2, 3, 4"
echo "  2. Execute 5-Fold ensemble"
echo "  3. Generate final submission"
