#!/bin/bash
# K-Fold Prediction Script
# Runs prediction for each fold and creates ensemble

set -e

export WANDB_API_KEY="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"

cd /data/ephemeral/home/baseline_code

echo "======================================================================="
echo "K-FOLD PREDICTION & ENSEMBLE"
echo "======================================================================="
echo ""

mkdir -p predictions/kfold

# Run prediction for each fold
for fold in 0 1 2 3 4; do
    echo ""
    echo "======================================================================="
    echo "Running Prediction for Fold ${fold}"
    echo "======================================================================="
    
    # Find best checkpoint for this fold
    CHECKPOINT=$(ls -t checkpoints/kfold/fold_${fold}/*.ckpt 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT" ]; then
        echo "⚠️  Warning: No checkpoint found for Fold ${fold}"
        echo "   Skipping this fold..."
        continue
    fi
    
    echo "Using checkpoint: $CHECKPOINT"
    
    # Run prediction
    python runners/predict.py \
        preset=example \
        checkpoint_path="'$CHECKPOINT'" \
        submission_dir="predictions/kfold" \
        2>&1 | tee "logs/fold${fold}_prediction.log"
    
    # Rename prediction file
    LATEST_PRED=$(ls -t predictions/kfold/*.json | head -1)
    if [ -f "$LATEST_PRED" ]; then
        mv "$LATEST_PRED" "predictions/kfold/fold${fold}_predictions.json"
        echo "✓ Saved: predictions/kfold/fold${fold}_predictions.json"
    fi
done

echo ""
echo "======================================================================="
echo "Creating Ensemble Prediction"
echo "======================================================================="

# Run ensemble
python scripts/ensemble_kfold.py

# Convert to CSV
ENSEMBLE_JSON="predictions/kfold_ensemble.json"
if [ -f "$ENSEMBLE_JSON" ]; then
    python ocr/utils/convert_submission.py "$ENSEMBLE_JSON"
    echo ""
    echo "✓ K-Fold ensemble complete!"
    echo "  JSON: $ENSEMBLE_JSON"
    echo "  CSV:  ${ENSEMBLE_JSON%.json}.csv"
fi

echo ""
echo "======================================================================="
echo "✓ ALL PREDICTIONS COMPLETE!"
echo "Submit: predictions/kfold_ensemble.csv"
echo "======================================================================="
