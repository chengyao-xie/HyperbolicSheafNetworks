#!/bin/bash

#  causality  correlation
python exp/train.py \
    --dataset abide \
    --construction_method pearson \
    --simplex_order 1 \
    --seed 42 \
    --epochs 60 \
    --lr 0.003 \
    --weight_decay 5e-4 \
    --hidden_channels 64 \
    --d 2 \
    --max_t 0.1 \
    --step_size 0.1 \
    --curvature 1.0 \
    --learn_curvature True \
    --input_dropout 0.5 \
    --dropout 0.5 \
    --pool_type attention \
    --early_stopping 20 \
    --grad_clip 1.0 \
    --amp auto \
    --eval_interval 1 \
    --log_interval 1 \
    --scheduler cosine \
    --checkpoint_metric accuracy \
    --save_model False \
    --num_workers 0 \
    --pin_memory True \
    --persistent_workers True \
    --accumulation_steps 1
