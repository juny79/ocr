#!/bin/bash

# Run 4 additional postprocessing tests
# Using checkpoint without '=' in filename

set -e

CKPT="outputs/efficientnet_b4_single/checkpoints/efficientnet_b4_epoch15.ckpt"
OUTPUT_BASE="outputs/efficientnet_b4_postproc_final/submissions"

mkdir -p ${OUTPUT_BASE}

echo "=================================================="
echo "Postprocessing 추가 테스트 (4가지)"
echo "=================================================="
echo "Checkpoint: ${CKPT}"
echo ""

# Test 1/4
echo "[1/4] thresh=0.29, box_thresh=0.25"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=test_t029_b025 \
  checkpoint_path=${CKPT} \
  models.head.postprocess.thresh=0.29 \
  models.head.postprocess.box_thresh=0.25 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/test_t029_b025/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_BASE}/submission_t0.29_b0.25.csv
echo "✅ submission_t0.29_b0.25.csv"
echo ""

# Test 2/4
echo "[2/4] thresh=0.29, box_thresh=0.24"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=test_t029_b024 \
  checkpoint_path=${CKPT} \
  models.head.postprocess.thresh=0.29 \
  models.head.postprocess.box_thresh=0.24 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/test_t029_b024/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_BASE}/submission_t0.29_b0.24.csv
echo "✅ submission_t0.29_b0.24.csv"
echo ""

# Test 3/4
echo "[3/4] thresh=0.30, box_thresh=0.25"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=test_t030_b025 \
  checkpoint_path=${CKPT} \
  models.head.postprocess.thresh=0.30 \
  models.head.postprocess.box_thresh=0.25 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/test_t030_b025/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_BASE}/submission_t0.30_b0.25.csv
echo "✅ submission_t0.30_b0.25.csv"
echo ""

# Test 4/4
echo "[4/4] thresh=0.30, box_thresh=0.26"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=test_t030_b026 \
  checkpoint_path=${CKPT} \
  models.head.postprocess.thresh=0.30 \
  models.head.postprocess.box_thresh=0.26 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/test_t030_b026/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_BASE}/submission_t0.30_b0.26.csv
echo "✅ submission_t0.30_b0.26.csv"
echo ""

echo "=================================================="
echo "✅ 모든 테스트 완료!"
echo "=================================================="
echo ""
echo "생성된 파일:"
ls -lh ${OUTPUT_BASE}/submission_t0.*.csv
echo ""
echo "📤 다음 단계: 리더보드에 제출하여 최적값 확인"
echo ""
echo "최적 파라미터 확인 후:"
echo "1. Sweep config 업데이트"
echo "2. WandB Sweep 재시작"
