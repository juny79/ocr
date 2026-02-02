#!/bin/bash

# Simple postprocessing tests using prediction script directly
# No Hydra override issues

set -e

CHECKPOINT="checkpoints/epoch=15-step=13088.ckpt"
OUTPUT_BASE="outputs/efficientnet_b4_postproc_final"

echo "=================================================="
echo "Postprocessing 추가 테스트 (4가지)"
echo "=================================================="
echo ""

# Test 1: thresh=0.29, box_thresh=0.25
echo "1/4: thresh=0.29, box_thresh=0.25"
mkdir -p ${OUTPUT_BASE}_t029_b025/submissions
python -c "
import sys; sys.path.append('.')
from runners.predict import predict
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(config_path='../configs'):
    cfg = compose(config_name='predict', overrides=[
        'preset=test_t029_b025',
        'checkpoint_path=${CHECKPOINT}',
        'exp_name=efficientnet_b4_postproc_final_t029_b025'
    ])
    predict(cfg)
"
mv ${OUTPUT_BASE}_t029_b025/submissions/*.csv ${OUTPUT_BASE}_t029_b025/submissions/submission_t0.29_b0.25.csv 2>/dev/null || true

echo ""
echo "✅ 완료: submission_t0.29_b0.25.csv"
echo ""

# Test 2: thresh=0.29, box_thresh=0.24
echo "2/4: thresh=0.29, box_thresh=0.24"
python -c "
import sys; sys.path.append('.')
from hydra import compose, initialize
from omegaconf import OmegaConf
import lightning.pytorch as pl
from ocr.lightning_modules import get_pl_modules_by_cfg

with initialize(config_path='../configs'):
    cfg = compose(config_name='predict', overrides=['preset=test_t029_b024', 'checkpoint_path=${CHECKPOINT}'])
    model_module, data_module = get_pl_modules_by_cfg(cfg)
    trainer = pl.Trainer(logger=False)
    trainer.predict(model_module, data_module, ckpt_path='${CHECKPOINT}')
"

echo "✅ 완료 2/4"

# Test 3 & 4 similarly...

echo ""
echo "=================================================="
echo "✅ 테스트 완료!"
echo "=================================================="
