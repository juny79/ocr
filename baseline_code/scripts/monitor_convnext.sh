#!/bin/bash
# ConvNeXt 학습 모니터링 스크립트

LOG_FILE="/data/ephemeral/home/baseline_code/convnext_tiny_training.log"

echo "================================================================================"
echo "📊 ConvNeXt-Tiny Training Monitor"
echo "================================================================================"
echo ""

# Check if process is running
if pgrep -f "train_convnext_tiny.py" > /dev/null; then
    echo "✅ Training process is RUNNING"
    PID=$(pgrep -f "train_convnext_tiny.py")
    echo "   PID: $PID"
else
    echo "⚠️  Training process NOT FOUND"
fi

echo ""
echo "📝 Recent Log (last 30 lines):"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE" | grep -E "Epoch|val/|Training|Complete" || tail -30 "$LOG_FILE"
else
    echo "❌ Log file not found: $LOG_FILE"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"

# Show validation metrics if available
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "📈 Validation Metrics (if available):"
    grep "val/hmean" "$LOG_FILE" | tail -5 || echo "   No validation metrics yet"
fi

echo ""
echo "💡 Commands:"
echo "   Monitor live:  tail -f $LOG_FILE"
echo "   Kill process:  pkill -f train_convnext_tiny.py"
echo "   Full log:      cat $LOG_FILE"
echo ""
