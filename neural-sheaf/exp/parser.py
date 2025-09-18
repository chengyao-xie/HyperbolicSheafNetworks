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
    parser = argparse.ArgumentParser()
    # Optimisation params
    parser.add_argument('--epochs', type=int, default=300)              
    parser.add_argument('--lr', type=float, default=0.001)              
    parser.add_argument('--weight_decay', type=float, default=1e-3)     
    parser.add_argument('--sheaf_decay', type=float, default=None)      
    parser.add_argument('--early_stopping', type=int, default=50)       
    parser.add_argument('--min_acc', type=float, default=0.0,
                        help="Minimum test acc on the first fold to continue training.")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='acc')

    # Model configuration
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--normalised', dest='normalised', type=str2bool, default=True,
                        help="Use a normalised Laplacian")
    parser.add_argument('--deg_normalised', dest='deg_normalised', type=str2bool, default=False,
                        help="Use a a degree-normalised Laplacian")
    parser.add_argument('--linear', dest='linear', type=str2bool, default=False,
                        help="Whether to learn a new Laplacian at each step.")
    parser.add_argument('--hidden_channels', type=int, default=32)   # 小样本：减小模型容量
    parser.add_argument('--input_dropout', type=float, default=0.3)  # 增强输入正则
    parser.add_argument('--dropout', type=float, default=0.6)        # 更大的dropout以防过拟合
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--second_linear', dest='second_linear', type=str2bool, default=False)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=False)

    # new
    parser.add_argument('--residual', type=str2bool, default=True,
                        help="Whether to use residual connections")
    parser.add_argument('--warm_start', type=int, default=10,
                        help="Number of warm-up epochs (小样本时可设置较小值)")
    parser.add_argument('--grad_clip', type=float, default=0.5,
                        help="Gradient clipping value")
    parser.add_argument('--lr_patience', type=int, default=10,
                        help="Patience for learning rate scheduler (小样本时可缩短)")

    # 小样本模式（便于在运行脚本中显式使用）
    parser.add_argument('--few_shot', dest='few_shot', type=str2bool, default=False,
                        help="If True, script/runner may apply small-sample-specific heuristics.")

    # 训练/评估加速
    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help='Enable Automatic Mixed Precision for faster training on CUDA')
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs to reduce ODE forward calls during eval')

    # Experiment parameters
    parser.add_argument('--dataset', default='abide')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--model', type=str, default='HyperbolicSheafODE',
                        choices=['DiagSheaf', 'BundleSheaf', 'GeneralSheaf', 'DiagSheafODE',
                                'BundleSheafODE', 'GeneralSheafODE','HyperbolicSheafODE'])
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--evectors', type=int, default=0, help="Number of Laplacian PE eigenvectors to use.")
    
    # 图分类任务参数
    parser.add_argument('--task', type=str, default='graph_classification',
                        choices=['node_classification', 'graph_classification'],
                        help="Task type: node classification or graph classification")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of classes for graph classification (default: 2 for binary)")
    parser.add_argument('--pool_type', type=str, default='mean',
                        choices=['mean', 'max', 'sum', 'attention'],
                        help="Pooling method for graph classification")
    parser.add_argument('--graph_classification', type=str2bool, default=False,
                        help="Whether to use graph classification mode")

    # ODE args
    parser.add_argument('--max_t', type=float, default=1.0, help="Maximum integration time.")
    parser.add_argument('--int_method', type=str, default='hyperbolic_rk4',
                       choices=['hyperbolic_rk4', 'rk4', 'euler', 'midpoint'],
                       help='ODE integration method')
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="hyperbolic_rk4",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")
    # 求解器参数
    parser.add_argument('--solver', type=str, default='hyperbolic_rk4', 
                       help='ODE solver type')
    
    # Contrastive learning arguments
    parser.add_argument('--use_contrastive', type=str2bool, default=False,
                       help='Enable contrastive learning (True/False)')
    parser.add_argument('--contrastive_weight', type=float, default=0.3,
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive learning')
    parser.add_argument('--projection_dim', type=int, default=64,
                        help='Projection dimension for contrastive learning')

    return parser