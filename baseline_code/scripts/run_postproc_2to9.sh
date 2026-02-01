#!/bin/bash

# Smart Postprocessing Optimization - Automated
# Runs 2-9

set -e

cd /data/ephemeral/home/baseline_code
CKPT_PATH="outputs/efficientnet_b4_single/checkpoints/epoch\=15-step\=13088.ckpt"
OUTPUT_DIR="outputs/efficientnet_b4_postproc_optim/submissions"

echo "========================================="
echo "Continuing Smart Postprocessing (2-9)"
echo "========================================="
echo ""

# Run 2: thresh=0.26, box_thresh=0.25
echo "[2/9] thresh=0.26, box_thresh=0.25"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_2 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.26 \
  models.head.postprocess.box_thresh=0.25 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_2/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.26_b0.25.csv
echo "✅ [2/9] submission_t0.26_b0.25.csv"
echo ""

# Run 3: thresh=0.28, box_thresh=0.25
echo "[3/9] thresh=0.28, box_thresh=0.25"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_3 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.28 \
  models.head.postprocess.box_thresh=0.25 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_3/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.28_b0.25.csv
echo "✅ [3/9] submission_t0.28_b0.25.csv"
echo ""

# Run 4: thresh=0.24, box_thresh=0.28
echo "[4/9] thresh=0.24, box_thresh=0.28"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_4 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.24 \
  models.head.postprocess.box_thresh=0.28 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_4/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.24_b0.28.csv
echo "✅ [4/9] submission_t0.24_b0.28.csv"
echo ""

# Run 5: thresh=0.24, box_thresh=0.30
echo "[5/9] thresh=0.24, box_thresh=0.30"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_5 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.24 \
  models.head.postprocess.box_thresh=0.30 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_5/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.24_b0.30.csv
echo "✅ [5/9] submission_t0.24_b0.30.csv"
echo ""

# Run 6: thresh=0.26, box_thresh=0.28
echo "[6/9] thresh=0.26, box_thresh=0.28"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_6 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.26 \
  models.head.postprocess.box_thresh=0.28 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_6/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.26_b0.28.csv
echo "✅ [6/9] submission_t0.26_b0.28.csv"
echo ""

# Run 7: thresh=0.25, box_thresh=0.27 (Balanced)
echo "[7/9] thresh=0.25, box_thresh=0.27 (Balanced)"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_7 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.25 \
  models.head.postprocess.box_thresh=0.27 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_7/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.25_b0.27.csv
echo "✅ [7/9] submission_t0.25_b0.27.csv"
echo ""

# Run 8: thresh=0.23, box_thresh=0.26 (Conservative)
echo "[8/9] thresh=0.23, box_thresh=0.26 (Conservative)"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_8 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.23 \
  models.head.postprocess.box_thresh=0.26 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_8/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.23_b0.26.csv
echo "✅ [8/9] submission_t0.23_b0.26.csv"
echo ""

# Run 9: thresh=0.27, box_thresh=0.26 (Aggressive)
echo "[9/9] thresh=0.27, box_thresh=0.26 (Aggressive)"
python runners/predict.py \
  preset=efficientnet_b4_aggressive \
  exp_name=postproc_optim_9 \
  checkpoint_path=$CKPT_PATH \
  models.head.postprocess.thresh=0.27 \
  models.head.postprocess.box_thresh=0.26 \
  models.head.postprocess.max_candidates=600 \
  > /dev/null 2>&1

JSON_FILE=$(ls -t outputs/postproc_optim_9/submissions/*.json 2>/dev/null | head -1)
python ocr/utils/convert_submission.py -J ${JSON_FILE} -O ${OUTPUT_DIR}/submission_t0.27_b0.26.csv
echo "✅ [9/9] submission_t0.27_b0.26.csv"
echo ""

echo "========================================="
echo "✅ 모든 예측 파일 생성 완료!"
echo "========================================="
echo ""
ls -lh ${OUTPUT_DIR}/*.csv
echo ""
echo "총 9개 제출 파일이 생성되었습니다."
echo "위치: ${OUTPUT_DIR}/"
echo ""
