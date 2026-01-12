#!/bin/bash

python exp/train_brain_graph.py \
    --dataset abide \
    --graph_type correlation \
    --seed 42 \
    --repeats 3 \
    --epochs 120 \
    --lr 0.003 \
    --weight_decay 1e-4 \
    --hidden_channels 128 \
    --d 2 \
    --max_t 0.1 \
    --step_size 0.5 \
    --curvature 1.0 \
    --learn_curvature True \
    --input_dropout 0.4 \
    --dropout 0.4 \
    --pool_type attention \
    --early_stopping 50 \
    --grad_clip 1.0


