# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from distutils.util import strtobool
import argparse


def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')


def get_parser():
    parser = argparse.ArgumentParser(description='Hyperbolic Sheaf ODE for Brain Graph Classification')
    
    # ========== 优化器参数 ==========
    parser.add_argument('--epochs', type=int, default=300,  
                        help="Maximum training epochs")
    parser.add_argument('--lr', type=float, default=0.01,  
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4,  
                        help="Weight decay for regularization")
    parser.add_argument('--sheaf_decay', type=float, default=None,
                        help="Separate weight decay for sheaf learner (default: same as weight_decay)")
    parser.add_argument('--early_stopping', type=int, default=100,  
                        help="Patience for early stopping")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='acc',
                        help="Early stopping based on validation loss or accuracy")
    parser.add_argument('--warm_start', type=int, default=3,  
                        help="Number of warm-up epochs for learning rate")
    parser.add_argument('--grad_clip', type=float, default=2.0, 
                        help="Gradient clipping value")

    # ========== 模型结构参数 ==========
    parser.add_argument('--layers', type=int, default=2,
                        help="Number of ODE blocks/layers")
    parser.add_argument('--d', type=int, default=2,
                        help="Sheaf stalk dimension")
    parser.add_argument('--hidden_channels', type=int, default=128,  
                        help="Hidden dimension size")
    parser.add_argument('--input_dropout', type=float, default=0.1,  
                        help="Dropout rate for input layer")
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help="Dropout rate for hidden layers")
    parser.add_argument('--left_weights', type=str2bool, default=True,
                        help="Apply left linear layer in ODE")
    parser.add_argument('--right_weights', type=str2bool, default=True,
                        help="Apply right linear layer in ODE")
    parser.add_argument('--use_act', type=str2bool, default=True,
                        help="Use activation function in ODE")
    parser.add_argument('--second_linear', type=str2bool, default=False,
                        help="Add second linear layer before ODE")
    parser.add_argument('--add_lp', type=str2bool, default=False,
                        help="Add fixed low-pass filter in restriction maps")
    parser.add_argument('--add_hp', type=str2bool, default=False,
                        help="Add fixed high-pass filter in restriction maps")
    parser.add_argument('--residual', type=str2bool, default=True,
                        help="Use residual connections")
    parser.add_argument('--sheaf_act', type=str, default="tanh",
                        help="Activation function in sheaf learner")

    # ========== 图分类参数 ==========
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of classes (default: 2 for binary classification)")
    parser.add_argument('--pool_type', type=str, default='attention',
                        choices=['mean', 'max', 'sum', 'attention'],
                        help="Graph pooling method")

    # ========== ODE 求解器参数 ==========
    parser.add_argument('--max_t', type=float, default=0.5,  
                        help="Maximum ODE integration time")
    parser.add_argument('--int_method', type=str, default='hyperbolic_rk4',
                        choices=['hyperbolic_rk4', 'rk4', 'euler', 'midpoint'],
                        help="ODE integration method")
    parser.add_argument('--step_size', type=float, default=0.5,  
                        help="Step size for fixed-step ODE solvers")
    parser.add_argument('--adjoint', action='store_true',
                        help="Use adjoint method to reduce memory footprint")
    parser.add_argument('--adjoint_method', type=str, default='rk4',
                        choices=['rk4', 'euler', 'midpoint'],
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
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--cuda', type=int, default=0,
                        help="CUDA device ID")
    parser.add_argument('--folds', type=int, default=10,
                        help="Number of cross-validation folds")
    parser.add_argument('--model', type=str, default='HyperbolicSheafODE',
                        choices=['HyperbolicSheafODE'],
                        help="Model type (only HyperbolicSheafODE supported)")

    # ========== 训练加速 ==========
    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help="Enable Automatic Mixed Precision for faster training")

    # ========== 兼容性参数（sheaf_base需要） ==========
    parser.add_argument('--normalised', type=str2bool, default=True,
                        help="Use normalised Laplacian")
    parser.add_argument('--deg_normalised', type=str2bool, default=False,
                        help="Use degree-normalised Laplacian")
    parser.add_argument('--linear', type=str2bool, default=False,
                        help="If True, use fixed Laplacian; if False, learn new Laplacian at each step")
    parser.add_argument('--sparse_learner', type=str2bool, default=False,
                        help="Use sparse sheaf learner")
    parser.add_argument('--edge_weights', type=str2bool, default=True,
                        help="Learn edge weights")
    parser.add_argument('--orth', type=str, default='householder',
                        choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        help="Orthogonal parametrization method")

    return parser