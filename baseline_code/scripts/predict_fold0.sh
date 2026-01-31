#!/bin/bash

cd /data/ephemeral/home/baseline_code

python runners/predict.py \
    preset=augmented_v2 \
    exp_name=aug_v2_fold0_final \
    checkpoint_path=outputs/aug_v2_fold0/checkpoints/epoch=23-step=8832.ckpt
