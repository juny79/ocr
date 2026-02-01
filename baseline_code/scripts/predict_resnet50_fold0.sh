#!/bin/bash

# ResNet50 Prediction Script
BASE_DIR="/data/ephemeral/home/baseline_code"

echo "========================================"
echo "Starting ResNet50 Prediction"
echo "Time: $(date)"
echo "========================================"

# Find best checkpoint
CHECKPOINT=$(ls -t $BASE_DIR/outputs/resnet50_fold0/checkpoints/*.ckpt | head -1)
echo "Using checkpoint: $CHECKPOINT"

cd $BASE_DIR

# Run prediction
python runners/predict.py \
    preset=augmented_resnet50 \
    ++checkpoint_path=$CHECKPOINT

if [ $? -eq 0 ]; then
    echo "Prediction completed successfully"
    
    # Find generated JSON file
    JSON_FILE=$(ls -t $BASE_DIR/outputs/ocr_training/submissions/*.json | head -1)
    echo "Generated JSON: $JSON_FILE"
    
    # Convert to CSV
    python ocr/utils/convert_submission.py \
        -J $JSON_FILE \
        -O outputs/submission_resnet50_fold0.csv <<< "yes"
    
    if [ $? -eq 0 ]; then
        echo "========================================"
        echo "Submission file created successfully!"
        echo "File: outputs/submission_resnet50_fold0.csv"
        echo "Time: $(date)"
        echo "========================================"
    else
        echo "CSV conversion failed!"
        exit 1
    fi
else
    echo "Prediction failed!"
    exit 1
fi
