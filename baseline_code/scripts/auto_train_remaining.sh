#!/bin/bash
# Auto-train remaining folds sequentially

cd /data/ephemeral/home/baseline_code

echo "========================================="
echo "Auto K-Fold Sequential Training"
echo "========================================="

# Wait for current fold to complete and start next
wait_and_train() {
    local FOLD=$1
    local LOG_FILE="logs/fold${FOLD}_train.log"
    
    echo ""
    echo "Waiting for Fold $((FOLD - 1)) to complete..."
    
    # Wait until completion message appears in previous fold's log
    local PREV_LOG="logs/fold$((FOLD - 1))_train.log"
    while ! grep -q "training completed" "$PREV_LOG" 2>/dev/null; do
        sleep 60
        echo "  Still waiting... ($(date +%H:%M:%S))"
    done
    
    echo "✓ Fold $((FOLD - 1)) completed!"
    echo ""
    echo "Starting Fold ${FOLD}..."
    
    nohup bash scripts/train_single_fold.sh ${FOLD} > ${LOG_FILE} 2>&1 &
    
    echo "✓ Fold ${FOLD} started (PID: $!)"
    echo "  Log: ${LOG_FILE}"
}

# Train Folds 3 and 4 sequentially
wait_and_train 3
wait_and_train 4

echo ""
echo "========================================="
echo "All folds queued for training!"
echo "========================================="
