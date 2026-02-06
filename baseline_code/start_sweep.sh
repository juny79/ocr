#!/bin/bash

# WandB Sweep 빠른 시작 가이드

echo "=========================================="
echo "🚀 HRNet-W44 1280x1280 WandB Sweep 시작"
echo "=========================================="
echo ""

cd /data/ephemeral/home/baseline_code

# Step 1: Sweep 초기화
echo "Step 1️⃣  Sweep 설정 초기화 중..."
echo ""
echo "명령어:"
echo "  wandb sweep sweep_hrnet_w44_1280.yaml --project hrnet-w44-1280-sweep --entity juny79"
echo ""
echo "실행:"

SWEEP_OUTPUT=$(wandb sweep sweep_hrnet_w44_1280.yaml --project hrnet-w44-1280-sweep --entity juny79 2>&1)
echo "$SWEEP_OUTPUT"

# SWEEP ID 추출
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'juny79/hrnet-w44-1280-sweep/[a-zA-Z0-9]+' | head -1)

if [ -z "$SWEEP_ID" ]; then
    echo ""
    echo "❌ Sweep ID를 자동으로 추출하지 못했습니다."
    echo "위의 출력에서 'wandb agent' 다음의 ID를 사용하세요."
    echo ""
    echo "예시: wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz"
    exit 1
fi

echo ""
echo "✅ Sweep ID: $SWEEP_ID"
echo ""
echo "=========================================="
echo ""
echo "Step 2️⃣  에이전트 실행"
echo ""
echo "옵션 A: 병렬 실행 (8개 동시)"
echo "--------"
echo "  wandb agent $SWEEP_ID --count 8"
echo ""
echo "옵션 B: 단일 실행"
echo "--------"
echo "  wandb agent $SWEEP_ID"
echo ""
echo "옵션 C: 지속 실행 (무한반복, Ctrl+C로 중단)"
echo "--------"
echo "  wandb agent $SWEEP_ID --count infinite"
echo ""
echo "=========================================="
echo ""
echo "✨ 실시간 모니터링:"
echo "  https://wandb.ai/juny79/hrnet-w44-1280-sweep"
echo ""
echo "=========================================="
echo ""

# 사용자 선택
echo "어떤 방식으로 실행할까요? (A/B/C/X):"
read -p "선택 (기본값: A): " choice
choice=${choice:-A}

case $choice in
    A|a)
        echo ""
        echo "🚀 병렬 실행 시작 (8개 에이전트)..."
        echo ""
        wandb agent "$SWEEP_ID" --count 8
        ;;
    B|b)
        echo ""
        echo "🚀 단일 실행 시작..."
        echo ""
        wandb agent "$SWEEP_ID"
        ;;
    C|c)
        echo ""
        echo "🚀 지속 실행 시작 (무한반복)..."
        echo "Ctrl+C로 언제든 중단 가능"
        echo ""
        wandb agent "$SWEEP_ID" --count infinite
        ;;
    X|x)
        echo ""
        echo "취소되었습니다."
        echo "나중에 수동으로 실행하세요:"
        echo "  wandb agent $SWEEP_ID --count 8"
        ;;
    *)
        echo ""
        echo "유효하지 않은 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Sweep 실행 완료!"
echo "=========================================="
