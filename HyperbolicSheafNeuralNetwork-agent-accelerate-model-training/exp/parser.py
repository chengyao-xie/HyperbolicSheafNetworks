# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from distutils.util import strtobool
import argparse


# 执行 str2bool 相关功能
def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')


def get_parser():
    parser = argparse.ArgumentParser(description='Hyperbolic Sheaf ODE for Brain Graph Classification')
    
    # ========== 优化器参数==========
    parser.add_argument('--epochs', type=int, default=60,
                        help="Maximum training epochs")
    parser.add_argument('--lr', type=float, default=0.003,
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4,  
                        help="Weight decay for regularization")
    parser.add_argument('--sheaf_decay', type=float, default=None,
                        help="Separate weight decay for sheaf learner (default: same as weight_decay)")
    parser.add_argument('--early_stopping', type=int, default=20,
                        help="Patience for early stopping")
    parser.add_argument('--grad_clip', type=float, default=1.0, 
                        help="Gradient clipping value")
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help="Optimizer type")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for 5-fold cross validation")
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help="Gradient accumulation steps for larger effective batch size")
    parser.add_argument('--num_folds', type=int, default=5,
                        help="Number of folds for cross validation")
    parser.add_argument('--fold_limit', type=int, default=0,
                        help="Only run the first N folds; <=0 runs all folds")
    parser.add_argument('--amp', type=str, default='auto',
                        choices=['auto', 'off', 'fp16', 'bf16'],
                        help="Mixed precision mode; auto enables CUDA autocast when available")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="DataLoader worker processes. Use 0 on Windows if multiprocessing is unstable")
    parser.add_argument('--pin_memory', type=str2bool, default=True,
                        help="Use pinned host memory for CUDA DataLoader transfers")
    parser.add_argument('--persistent_workers', type=str2bool, default=True,
                        help="Keep DataLoader workers alive between epochs when num_workers > 0")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help="DataLoader prefetch factor when num_workers > 0")
    parser.add_argument('--eval_interval', type=int, default=1,
                        help="Evaluate every N epochs; final epoch is always evaluated")
    parser.add_argument('--log_interval', type=int, default=1,
                        help="Print one training line every N epochs")
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine', 'none'],
                        help="Learning-rate scheduler")
    parser.add_argument('--checkpoint_metric', type=str, default='accuracy',
                        choices=['accuracy', 'balanced_accuracy', 'auc', 'loss'],
                        help="Metric used for best checkpoint and early stopping")
    parser.add_argument('--save_model', type=str2bool, default=False,
                        help="Save the best model checkpoint; metrics are always recorded")
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help="Minimum checkpoint metric improvement for early stopping")
    parser.add_argument('--compile_model', type=str2bool, default=False,
                        help="Use torch.compile when available")
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help="torch.compile mode")
    parser.add_argument('--deterministic', type=str2bool, default=True,
                        help="Use deterministic cuDNN settings for reproducibility")
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False,
                        help="Enable cuDNN benchmark for fixed-size inputs")

    # ========== 模型结构参数 ==========
    parser.add_argument('--layers', type=int, default=2,
                        help="Number of ODE blocks/layers")
    parser.add_argument('--d', type=int, default=2,
                        help="Sheaf stalk dimension")
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help="Hidden dimension size")
    parser.add_argument('--input_dropout', type=float, default=0.5,  
                        help="Dropout rate for input layer")
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help="Dropout rate for hidden layers")
    parser.add_argument('--left_weights', type=str2bool, default=True,
                        help="Apply left linear layer in ODE")
    parser.add_argument('--right_weights', type=str2bool, default=True,
                        help="Apply right linear layer in ODE")
    parser.add_argument('--use_act', type=str2bool, default=True,
                        help="Use activation function in ODE")
    parser.add_argument('--second_linear', type=str2bool, default=False,
                        help="Add second linear layer before ODE")
    parser.add_argument('--sheaf_act', type=str, default="tanh",
                        help="Activation function in sheaf learner")

    # ========== 图分类参数==========
    parser.add_argument('--pool_type', type=str, default='attention',
                        choices=['mean', 'max', 'sum', 'attention'],
                        help="Graph pooling method")

    # ========== ODE 求解器参数==========
    parser.add_argument('--max_t', type=float, default=0.1,  
                        help="Maximum ODE integration time")
    parser.add_argument('--int_method', type=str, default='hyperbolic_rk4',
                        choices=['hyperbolic_rk4', 'rk4', 'euler'],
                        help="ODE integration method")
    parser.add_argument('--step_size', type=float, default=0.1,
                        help="Step size for fixed-step ODE solvers")
    parser.add_argument('--curvature', type=float, default=1.0,
                        help="Initial/Fixed curvature c of Poincaré ball; set 0 for Euclidean")
    parser.add_argument('--learn_curvature', type=str2bool, default=False,
                        help="Whether to learn curvature as a parameter")
    parser.add_argument('--adjoint', action='store_true',
                        help="Use adjoint method to reduce memory footprint")
    parser.add_argument('--adjoint_method', type=str, default='rk4',
                        choices=['rk4', 'euler'],
                        help="ODE solver for backward pass")
    parser.add_argument('--adjoint_step_size', type=float, default=1.0,
                        help="Step size for adjoint solver")
    parser.add_argument('--tol_scale', type=float, default=1.0,
                        help="Multiplier for atol and rtol")
    parser.add_argument('--tol_scale_adjoint', type=float, default=1.0,
                        help="Multiplier for adjoint_atol and adjoint_rtol")

    # ========== 实验参数 ==========
    parser.add_argument('--dataset', type=str, default='abide',
                        help="Dataset name: abide, ppmi, etc.")
    parser.add_argument('--simplex_order', type=int, default=2, choices=[1, 2],
                        help="Order of simplicial complex: 1 (nodes+edges) or 2 (nodes+edges+triangles)")
    parser.add_argument('--top_k_percent', type=int, default=20,
                        help="Keep top k percent strongest graph edges")
    parser.add_argument('--max_triangles', type=int, default=500,
                        help="Maximum number of triangles per graph; <=0 means unlimited")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument('--cuda', type=int, default=0,
                        help="CUDA device ID")
    parser.add_argument('--max_nfe', type=int, default=5000,
                        help="Maximum number of function evaluations for ODE solver")

    # ========== 兼容性参数（sheaf_base）==========
    parser.add_argument('--normalised', type=str2bool, default=True,
                        help="Use normalised Laplacian")
    parser.add_argument('--deg_normalised', type=str2bool, default=False,
                        help="Use degree-normalised Laplacian")
    parser.add_argument('--construction_method', type=str, default='pearson',
                        choices=['pearson', 'causality'],
                        help="Brain graph construction method")
    parser.add_argument('--cs_embedding_dim', type=int, default=3,
                        help="Takens embedding dimension for continuity scaling")
    parser.add_argument('--cs_delay', type=int, default=1,
                        help="Takens embedding delay for continuity scaling")
    parser.add_argument('--cs_scales', type=int, default=33,
                        help="Number of scales for continuity scaling")
    parser.add_argument('--cs_theiler_window', type=int, default=2,
                        help="Temporal exclusion window for continuity scaling")
    parser.add_argument('--cs_epsilon_min', type=float, default=0.001,
                        help="Minimum epsilon percent for continuity scaling")
    parser.add_argument('--cs_epsilon_max', type=float, default=1.0,
                        help="Maximum epsilon percent for continuity scaling")
    parser.add_argument('--preprocess_workers', type=int, default=0,
                        help="Parallel workers for CPU preprocessing; 0 keeps serial preprocessing")
    parser.add_argument('--use_learned_sheaf', type=str2bool, default=True,
                        help="Use learnable sheaf Laplacian in diffusion")
    parser.add_argument('--use_fixed_hodge', type=str2bool, default=True,
                        help="Use fixed Hodge Laplacian correction for order-2 data")
    parser.add_argument('--hodge_weight', type=float, default=1.0,
                        help="Weight of fixed Hodge correction")
    return parser
