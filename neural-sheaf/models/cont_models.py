# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
import numpy as np
from torch_scatter import scatter_add

from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner
# from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint
from lib.adjoint import odeint, odeint_adjoint
from lib.hyperbolic_geometry import PoincareBall


class HyperbolicLaplacianODEFunc(nn.Module):
    """Implements hyperbolic diffusion based on Equation 15 using PoincareBall."""
    def __init__(self,
                d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels,
                left_weights=False,
                right_weights=False,
                use_act=False,
                nonlinear=False,
                weight_learner=None):

        super(HyperbolicLaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner  # Optional learned diffusivity
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.nonlinear = nonlinear
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        self.nfe = 0
        self.gamma = nn.Parameter(torch.tensor(0.01))  # 更小的初始值
        
        self.L = None
        self.edge_weights = None  # New: initialize edge_weights
        
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

        # self.beta = 1.0  # Parameter for distance-based diffusivity
        # self.sigma = nn.ReLU()  # Activation function
        self.c = 1.0  # Curvature of Poincaré ball
        self.manifold = PoincareBall()  # Trustworthy Poincaré ball implementation

    def update_laplacian_builder(self, laplacian_builder):
        self.edge_index = laplacian_builder.edge_index
        self.laplacian_builder = laplacian_builder
        
    def forward(self, t, x):
        if self.nonlinear or self.L is None:
            # Update the laplacian at each step.
            x_maps = x.view(self.graph_size, -1)
            maps = self.sheaf_learner(x_maps, self.edge_index)
            if self.weight_learner is not None:
                self.edge_weights = self.weight_learner(x_maps, self.edge_index)  # Save edge_weights
                L, _ = self.laplacian_builder(maps, self.edge_weights)
            else:
                L, _ = self.laplacian_builder(maps)
            self.L = L
        else:
            # Cache the Laplacian obtained at the first layer for the rest of the integration.
            L = self.L

        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, self.graph_size * self.d).t()

        if self.right_weights:
            x = self.lin_right_weights(x)

            
            
        # if self.nfe > self.opt["max_nfe"]:
        #     raise RuntimeError(f"Exceeded max NFE (got {self.nfe}, max allowed {self.opt['max_nfe']})")
        self.nfe += 1

        # Ensure shape (N, d)
        if x.dim() == 2 and x.size(0) == self.graph_size:
            z = x
        else:
            z = x.view(self.graph_size, -1)

        # Take embeddings from both ends of the edge
        src, dst = self.edge_index[0], self.edge_index[1]
        z_src = z[src]  # (E, d)
        z_dst = z[dst]  # (E, d)

        # Compute log map in hyperbolic space: log_{z_src}(z_dst)
        logmaps = self.manifold.logmap(z_src, z_dst, c=self.c)  # (E, d)
        logmaps = torch.clamp(logmaps, min=-10.0, max=10.0)

        # Weights
        # Ensure edge_weights exist
        if self.edge_weights is None:
            self.edge_weights = torch.ones(self.edge_index.size(1), device=x.device)
            
        ew = self.edge_weights.unsqueeze(-1)  # (E, 1)
        ew = torch.clamp(ew, min=0.0, max=10.0)
        weighted = ew * logmaps  # (E, d)

        # Aggregate to source nodes
        v = scatter_add(weighted, src, dim=0, dim_size=self.graph_size)  # (N, d)

        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))

        # if self.use_act:
        #     # x = F.elu(x)
        #     v = self.sigma(v) # Activation function, usually not needed

        # v = self.manifold.proj_tan(v, c=self.c)

        # Apply gamma scaling factor with更强的约束
        v = torch.clamp(self.gamma, min=0.001, max=0.5) * v
        
        # After evolution in tangent space, map back to hyperbolic space
        v = self.manifold.expmap(z, v, c=self.c)
        # Ensure output is on the hyperbolic disk
        v = self.manifold.proj(v, c=self.c)

        # 最终的数值稳定性检查
        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        v = torch.clamp(v, min=-5, max=5)
        
        # Ensure output shape matches input shape
        v = v.view(-1)  # Reshape to (graph_size * d,)

        
        return v # Shape: (graph_size * d,)

class LaplacianODEFunc(nn.Module):
    """Implements Laplacian-based diffusion."""

    def __init__(self,
                 d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels,
                 left_weights=False,
                 right_weights=False,
                 use_act=False,
                 nonlinear=False,
                 weight_learner=None):
        """
        Args:
            L: A sparse Laplacian matrix.
        """
        super(LaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.nonlinear = nonlinear
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        self.L = None

        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

    def update_laplacian_builder(self, laplacian_builder):
        self.edge_index = laplacian_builder.edge_index
        self.laplacian_builder = laplacian_builder

    def forward(self, t, x):
        if self.nonlinear or self.L is None:
            # Update the laplacian at each step.
            x_maps = x.view(self.graph_size, -1)
            maps = self.sheaf_learner(x_maps, self.edge_index)
            if self.weight_learner is not None:
                edge_weights = self.weight_learner(x_maps, self.edge_index)
                L, _ = self.laplacian_builder(maps, edge_weights)
            else:
                L, _ = self.laplacian_builder(maps)
            self.L = L
        else:
            # Cache the Laplacian obtained at the first layer for the rest of the integration.
            L = self.L

        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, self.graph_size * self.d).t()

        if self.right_weights:
            x = self.lin_right_weights(x)

        x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), -x)

        if self.use_act:
            x = F.elu(x)

        return x


class ODEBlock(nn.Module):
    """Module performing the ODE Integration."""

    def __init__(self, odefunc, t, opt):
        super(ODEBlock, self).__init__()
        self.t = t
        self.opt = opt
        self.odefunc = odefunc
        self.set_tol()
        self.manifold = PoincareBall()

    def set_tol(self):
        self.atol = self.opt['tol_scale'] * 1e-7
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint']:
            self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def forward(self, x):
        # Ensure using supported solver method
        int_method = self.opt.get('int_method', 'hyperbolic_rk4')
        if int_method == 'dopri5':
            print("[WARNING] Dopri5 solver is not implemented, switching to hyperbolic_rk4")
            int_method = 'hyperbolic_rk4'
            
        adjoint_method = self.opt.get('adjoint_method', 'rk4')
        if adjoint_method == 'dopri5':
            print("[WARNING] Dopri5 adjoint solver is not implemented, switching to rk4")
            adjoint_method = 'rk4'

        # 仅为定步长方法传入支持的参数：step_size、manifold；移除 max_iters
        options = dict(step_size=self.opt['step_size'], manifold=self.manifold)
        adjoint_options = dict(step_size=self.opt['adjoint_step_size'], manifold=self.manifold)

        if self.opt["adjoint"] and self.training:
            print('111111111111111')
            z = odeint_adjoint(
                self.odefunc, x, self.t,
                method=int_method,
                options=options,
                adjoint_method=adjoint_method,
                adjoint_options=adjoint_options,
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint)
        else:
            #print('222222222222')
            # print('odeint training')
            z = odeint(
                self.odefunc, x, self.t,
                method=int_method,
                options=options,
                atol=self.atol,
                rtol=self.rtol)
        self.odefunc.L = None
        # For fixed step solvers, we need to extract the last element
        if isinstance(z, torch.Tensor):
            z = z[-1]
        else:
            z = z[1]
        return z


class HyperbolicGraphDiffusion(SheafDiffusion):
    """Diffusion model using hyperbolic Laplacian ODE for graph classification."""

    def __init__(self, edge_index, args):
        super(HyperbolicGraphDiffusion, self).__init__(edge_index, args)
        # Hyperbolic dimension
        # self.d = args.get('hyperbolic_dim', 2)  # defined in the SheafDiffusion class
        self.c = 1.0 # hyperbolic curvature
        print("input_dim:", self.input_dim)
        
        # Check if it's a graph classification task
        self.graph_classification = args.get('graph_classification', True)
        
        if not hasattr(self, 'int_method'):
            self.int_method = 'hyperbolic_rk4'
    
        # Input projection to hyperbolic space
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
            
        # Modify output layer for graph classification
        if self.graph_classification:
            # Graph-level classification: pool first then classify
            self.pool_type = args.get('pool_type', 'mean')  # 'mean', 'max', 'sum', 'attention'
            
            # 注意力池化层（可选）
            if self.pool_type == 'attention':
                self.attention_pool = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, 1)
                )
            
            # 图分类器
            self.graph_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
        else:
            # 原始节点分类
            self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, (self.d,))
        # self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
        #     self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
        #     add_lp=self.add_lp, orth_map=self.orth_trans)
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)


        # Define hyperbolic ODE function
        self.odefunc = HyperbolicLaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)
        self.manifold = PoincareBall()

        # 1. Add layer normalization
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
        # 2. Add residual connections
        self.residual = args.get('residual', True)
        
        # 3. Modify activation function
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        # 4. Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # 更保守的Xavier初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # 减小初始化规模
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)
        
        # Initialize gamma to a very small value
        with torch.no_grad():
            if hasattr(self.odefunc, 'gamma'):
                self.odefunc.gamma.fill_(0.01)  # 更小的初始值
                
    def graph_pooling(self, x, batch=None):
        """图级别池化操作"""
        if self.pool_type == 'mean':
            if batch is None:
                # 单图情况：对所有节点求平均
                return torch.mean(x, dim=0, keepdim=True)
            else:
                # 批处理情况：按batch分组求平均
                from torch_geometric.nn import global_mean_pool
                return global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            if batch is None:
                return torch.max(x, dim=0, keepdim=True)[0]
            else:
                from torch_geometric.nn import global_max_pool
                return global_max_pool(x, batch)
        elif self.pool_type == 'sum':
            if batch is None:
                return torch.sum(x, dim=0, keepdim=True)
            else:
                from torch_geometric.nn import global_add_pool
                return global_add_pool(x, batch)
        elif self.pool_type == 'attention':
            # 注意力池化
            attention_weights = self.attention_pool(x)  # (N, 1)
            attention_weights = F.softmax(attention_weights, dim=0)
            if batch is None:
                graph_embed = torch.sum(x * attention_weights, dim=0, keepdim=True)
            else:
                # 批处理的注意力池化需要按batch分组
                graph_embed = []
                for i in range(batch.max().item() + 1):
                    mask = (batch == i)
                    x_graph = x[mask]
                    att_graph = attention_weights[mask]
                    att_graph = F.softmax(att_graph, dim=0)
                    graph_embed.append(torch.sum(x_graph * att_graph, dim=0))
                graph_embed = torch.stack(graph_embed)
            return graph_embed
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")
                
    def forward(self, x, batch=None):
        # 1. Input processing
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        
        # 2. First transformation layer
        x = self.lin1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 3. 可选的第二个线性层
        if self.second_linear:
            x = self.lin12(x)
            x = self.layer_norm2(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # 4. 投影到双曲空间 - 更保守的映射
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)  # 归一化避免过大的值
        x = self.manifold.proj(
            self.manifold.expmap0(
                self.manifold.proj_tan0(x_normalized * 0.5, self.c),  # 缩放因子0.5
                c=self.c
            ), 
            c=self.c
        )
        
        # 5. ODE integration - 添加数值稳定性检查
        if self.t > 0:
            x_flat = x.view(-1)
            # 检查输入是否有效
            if torch.isnan(x_flat).any() or torch.isinf(x_flat).any():
                print(f"[WARNING] Invalid input to ODE: nan={torch.isnan(x_flat).any()}, inf={torch.isinf(x_flat).any()}")
                x_flat = torch.clamp(x_flat, min=-1, max=1)
            
            x_evolved = self.odeblock(x_flat)
            
            # 检查输出是否有效
            if torch.isnan(x_evolved).any() or torch.isinf(x_evolved).any():
                print(f"[WARNING] Invalid ODE output: nan={torch.isnan(x_evolved).any()}, inf={torch.isinf(x_evolved).any()}")
                x_evolved = torch.clamp(x_evolved, min=-5, max=5)
            
            x = x_evolved.view(self.graph_size, -1)
        
        # 6. Project back to Euclidean space
        x = self.manifold.logmap0(x, self.c)
        
        # 最终的数值稳定性检查
        x = torch.clamp(x, min=-10, max=10)
        
        # 7. 输出层
        if self.graph_classification:
            # Graph classification: pool first then classify
            graph_embed = self.graph_pooling(x, batch)  # (batch_size, hidden_dim)
            output = self.graph_classifier(graph_embed)  # (batch_size, output_dim)
            if self.output_dim == 2:
                # Binary classification: return single logit value (batch_size, 1)
                return output[:, 0:1]  # Take only first column as logit
            else:
                # 多分类：返回log_softmax
                return F.log_softmax(output, dim=1)
        else:
            # 节点分类
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.edge_index = edge_index
        if self.diffusivity_learner is not None:
            self.diffusivity_learner.update_edge_index(edge_index)


class GraphLaplacianDiffusion(SheafDiffusion):
    """This is a diffusion model based on the weighted graph Laplacian."""

    def __init__(self, edge_index, args):
        super(GraphLaplacianDiffusion, self).__init__(edge_index, args)
        assert args['d'] == 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = EdgeWeightLearner(self.hidden_dim, edge_index)
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        self.sheaf_learner.update_edge_index(edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiagSheafDiffusion(SheafDiffusion):
    """Performs diffusion using a sheaf Laplacian with diagonal restriction maps."""

    def __init__(self, edge_index, args):
        super(DiagSheafDiffusion, self).__init__(edge_index, args)

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class BundleSheafDiffusion(SheafDiffusion):
    """Performs diffusion using a sheaf Laplacian with diagonal restriction maps."""

    def __init__(self, edge_index, args):
        super(BundleSheafDiffusion, self).__init__(edge_index, args)
        # Should use diagonal sheaf diffusion instead if d=1.
        assert args['d'] > 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.weight_learner = EdgeWeightLearner(self.hidden_dim, edge_index) if self.use_edge_weights else None
        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.get_param_size(),),
                                                     sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, weight_learner=self.weight_learner)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        self.weight_learner.update_edge_index(edge_index)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(GeneralSheafDiffusion, self).__init__(edge_index, args)
        # Should use diagoal diffusion if d == 1
        assert args['d'] > 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class HyperbolicGraphClassifier(nn.Module):
    """
    Hyperbolic Graph Classifier for brain graph classification.
    Wraps HyperbolicGraphDiffusion for graph-level classification tasks.
    """
    
    def __init__(self, edge_index, args):
        super(HyperbolicGraphClassifier, self).__init__()
        self.args = args
        self.task = 'graph_classification'
        
        # 确保设置为图分类模式
        if isinstance(args, dict):
            args_copy = args.copy()
        else:
            args_copy = vars(args).copy()
            
        args_copy['graph_classification'] = True
        args_copy['num_classes'] = args_copy.get('num_classes', 2)
        
        # 创建超曲图扩散模型
        self.graph_diffusion = HyperbolicGraphDiffusion(edge_index, args_copy)
        
        # 损失函数
        self.num_classes = args_copy['num_classes']
        if self.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Contrastive learning related parameters
        self.use_contrastive = args_copy.get('use_contrastive', True)
        self.contrastive_weight = args_copy.get('contrastive_weight', 0.5)
        self.temperature = args_copy.get('temperature', 0.1)
        self.projection_dim = args_copy.get('projection_dim', 128)
        
        # Contrastive learning projection head
        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(self.graph_diffusion.hidden_dim, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim)
            )
            
        # 统计信息
        self.metrics_tracker = {}
        
        # 为对比学习启用epoch统计（需要在run.py中绑定相关函数后才能调用）
        # self.reset_epoch_stats(-1, 'start') # 保持注释状态，由run.py绑定后调用

    def forward(self, x, batch=None, return_embedding=False):
        """前向传播"""
        output = self.graph_diffusion(x, batch)
        
        if return_embedding and self.use_contrastive:
            # 获取图嵌入用于对比学习
            # 重新进行图池化获取嵌入
            x_processed = self.graph_diffusion.lin1(x)
            if self.graph_diffusion.second_linear:
                x_processed = self.graph_diffusion.lin12(x_processed)
            graph_embed = self.graph_diffusion.graph_pooling(x_processed, batch)
            projected_embed = self.projection_head(graph_embed)
            return output, projected_embed
        
        return output
    
    def contrastive_loss(self, embeddings_list, labels_list):
        """
        计算对比学习损失
        embeddings_list: 多个图的嵌入列表
        labels_list: 对应的标签列表
        """
        if len(embeddings_list) < 2:
            return torch.tensor(0.0, device=embeddings_list[0].device)
            
        # Merge all embeddings and labels
        all_embeddings = torch.cat(embeddings_list, dim=0)  # (total_graphs, projection_dim)
        # 确保 labels 至少为 1 维，再拼接
        safe_labels = []
        for lbl in labels_list:
            if lbl.dim() == 0:
                lbl = lbl.unsqueeze(0)
            else:
                lbl = lbl.view(-1)
            safe_labels.append(lbl)
        all_labels = torch.cat(safe_labels, dim=0)  # (total_graphs,)
        
        # Normalize embeddings
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(all_embeddings, all_embeddings.t()) / self.temperature
        
        # Create positive sample mask (same labels)
        labels_expanded = all_labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()).float()
        
        # Remove diagonal elements (similarity with self)
        pos_mask.fill_diagonal_(0)
        
        # Create negative sample mask
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Compute loss for each sample
        total_loss = 0
        num_positives = 0
        
        for i in range(len(all_labels)):
            # Positive samples
            pos_indices = torch.where(pos_mask[i] > 0)[0]
            if len(pos_indices) == 0:
                continue
                
            # Negative samples
            neg_indices = torch.where(neg_mask[i] > 0)[0]
            
            # Compute denominator: positive samples + negative samples
            pos_sum = exp_sim[i, pos_indices].sum()
            neg_sum = exp_sim[i, neg_indices].sum()
            
            # InfoNCE loss
            if pos_sum > 0 and (pos_sum + neg_sum) > 0:
                loss_i = -torch.log(pos_sum / (pos_sum + neg_sum))
                total_loss += loss_i
                num_positives += 1
        
        if num_positives > 0:
            return total_loss / num_positives
        else:
            return torch.tensor(0.0, device=all_embeddings.device)

    def compute_metrics_multiple(self, embeddings_list, data_list, split):
        """
        计算多个图的评估指标，整合对比学习
        """
        if not isinstance(embeddings_list, list):
            embeddings_list = [embeddings_list]
        if not isinstance(data_list, list):
            data_list = [data_list]
            
        logits_list = []
        labels_list = []
        embeddings_for_contrastive = []
        total_loss = 0
        total_contrastive_loss = 0
        
        for i, (embeddings, data) in enumerate(zip(embeddings_list, data_list)):
            # Forward propagation
            if self.use_contrastive:
                logits, projected_embed = self.forward(embeddings, return_embedding=True)
                embeddings_for_contrastive.append(projected_embed)
            else:
                logits = self.forward(embeddings)
            
            # 获取标签
            if 'graph_labels' in data:
                labels = data['graph_labels']
            elif 'y' in data:
                labels = data['y']
            elif 'labels' in data:
                labels = data['labels']
            else:
                raise KeyError("No graph labels found in data")
                
            # 统一 labels 形状为 [N]（通常为 [1]）
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            else:
                labels = labels.view(-1)
                
            # Ensure labels are on the correct device
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            
            logits_list.append(logits)
            labels_list.append(labels)
            
            # 计算分类损失（统一展平为 1D，避免 squeeze 变成 0D）
            if self.num_classes == 2:
                logits_1d = logits.view(-1)
                labels_1d = labels.view(-1).float()
                loss = self.criterion(logits_1d, labels_1d)
            else:
                loss = self.criterion(logits, labels.long())
                
            total_loss += loss
        
        # Compute contrastive learning loss
        if self.use_contrastive and len(embeddings_for_contrastive) > 1:
            contrastive_loss = self.contrastive_loss(embeddings_for_contrastive, labels_list)
            total_contrastive_loss = contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=embeddings_list[0].device if len(embeddings_list) > 0 else None)
            total_contrastive_loss = contrastive_loss

        # 对分类损失取平均，再与对比损失组合
        num_graphs = max(1, len(logits_list))
        avg_classification_loss = total_loss / num_graphs
        combined_loss = avg_classification_loss + self.contrastive_weight * contrastive_loss if self.use_contrastive else avg_classification_loss

        # Merge all predictions and labels for evaluation
        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        # Compute predictions and accuracy
        if self.num_classes == 2:
            # 避免 squeeze 导致 0 维
            all_logits = all_logits.view(-1)
            probs = torch.sigmoid(all_logits)
            preds = (probs > 0.5).float()
            labels_for_acc = all_labels.view(-1).float()
        else:
            probs = F.softmax(all_logits, dim=1)
            preds = torch.argmax(all_logits, dim=1)
            labels_for_acc = all_labels.view(-1).long()
        
        acc = (preds == labels_for_acc).float().mean()
        
        # Compute other metrics
        if self.num_classes == 2:
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                labels_np = all_labels.view(-1).detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()
                
                if len(np.unique(labels_np)) > 1:
                    roc = roc_auc_score(labels_np, probs_np)
                    ap = average_precision_score(labels_np, probs_np)
                else:
                    roc = 0.0
                    ap = 0.0
            except Exception as e:
                print(f"Warning: Could not compute ROC/AP metrics: {e}")
                roc = 0.0
                ap = 0.0
        else:
            roc = 0.0
            ap = 0.0
        
        metrics = {
            'loss': combined_loss,
            'classification_loss': avg_classification_loss,
            'contrastive_loss': total_contrastive_loss,
            'acc': acc.item(),
            'roc': roc,
            'ap': ap,
            'num_graphs': len(data_list)
        }
        
        return metrics
    
    def decode(self, x, batch=None):
        return self.forward(x, batch)
        
    @property 
    def sheaf_learners(self):
        # 获取sheaf_learners
        return getattr(self.graph_diffusion, 'sheaf_learners', [])
    
    @property
    def epsilons(self):
        # 获取epsilons
        return getattr(self.graph_diffusion, 'epsilons', [])

    def update_edge_index(self, edge_index):
        # 更新边索引
        self.graph_diffusion.update_edge_index(edge_index)
        
    def grouped_parameters(self):
        # 返回分组的参数，用于优化器
        return self.graph_diffusion.grouped_parameters()




class EuclideanGraphDiffusion(SheafDiffusion):
    """欧氏空间图扩散（消融：用欧氏 ODE 替换双曲 ODE）"""

    def __init__(self, edge_index, args):
        super(EuclideanGraphDiffusion, self).__init__(edge_index, args)
        # 设定为常规 RK4（避免使用 hyperbolic_rk4）
        if isinstance(args, dict):
            args.setdefault('int_method', 'rk4')
        else:
            # argparse.Namespace 的情况
            if not hasattr(args, 'int_method'):
                setattr(args, 'int_method', 'rk4')

        # 图/节点分类模式
        self.graph_classification = args.get('graph_classification', True)

        # 前端特征变换
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 图分类头
        if self.graph_classification:
            self.pool_type = args.get('pool_type', 'mean')
            if self.pool_type == 'attention':
                self.attention_pool = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, 1)
                )
            self.graph_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
        else:
            self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # 与双曲版保持一致的 sheaf 与拉普拉斯构造（只是 ODE 换成欧氏）
        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, (self.d,), sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)

        # 关键：使用欧氏 ODE
        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index,
            self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights,
            right_weights=self.right_weights, use_act=self.use_act)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

        # 与双曲版一致的层归一化/激活（公平消融）
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.residual = args.get('residual', True)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        if self.graph_classification:
            for layer in self.graph_classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.xavier_uniform_(self.lin2.weight)
        if self.second_linear:
            nn.init.xavier_uniform_(self.lin12.weight)

    def graph_pooling(self, x, batch=None):
        if self.pool_type == 'mean':
            if batch is None:
                return torch.mean(x, dim=0, keepdim=True)
            else:
                from torch_geometric.nn import global_mean_pool
                return global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            if batch is None:
                return torch.max(x, dim=0, keepdim=True)[0]
            else:
                from torch_geometric.nn import global_max_pool
                return global_max_pool(x, batch)
        elif self.pool_type == 'sum':
            if batch is None:
                return torch.sum(x, dim=0, keepdim=True)
            else:
                from torch_geometric.nn import global_add_pool
                return global_add_pool(x, batch)
        elif self.pool_type == 'attention':
            att = self.attention_pool(x)
            att = F.softmax(att, dim=0)
            if batch is None:
                return torch.sum(x * att, dim=0, keepdim=True)
            else:
                graph_embed = []
                for i in range(batch.max().item() + 1):
                    mask = (batch == i)
                    xg = x[mask]
                    attg = F.softmax(att[mask], dim=0)
                    graph_embed.append(torch.sum(xg * attg, dim=0))
                return torch.stack(graph_embed)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")

    def forward(self, x, batch=None):
        # 1) 前端特征
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
            x = self.layer_norm2(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2) 欧氏 ODE 演化（无双曲映射）
        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)

        # 3) 输出
        if self.graph_classification:
            graph_embed = self.graph_pooling(x, batch)
            output = self.graph_classifier(graph_embed)
            if self.output_dim == 2:
                return output[:, 0:1]
            else:
                return F.log_softmax(output, dim=1)
        else:
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)


class EuclideanGraphClassifier(nn.Module):
    """欧氏空间图分类器（与 HyperbolicGraphClassifier 同接口，便于消融对比）"""
    def __init__(self, edge_index, args):
        super(EuclideanGraphClassifier, self).__init__()
        self.args = args
        self.task = 'graph_classification'

        if isinstance(args, dict):
            args_copy = args.copy()
        else:
            args_copy = vars(args).copy()
        args_copy['graph_classification'] = True
        args_copy['num_classes'] = args_copy.get('num_classes', 2)
        args_copy.setdefault('int_method', 'rk4')

        self.graph_diffusion = EuclideanGraphDiffusion(edge_index, args_copy)

        self.num_classes = args_copy['num_classes']
        self.criterion = nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()

        self.use_contrastive = args_copy.get('use_contrastive', True)
        self.contrastive_weight = args_copy.get('contrastive_weight', 0.5)
        self.temperature = args_copy.get('temperature', 0.1)
        self.projection_dim = args_copy.get('projection_dim', 128)

        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(self.graph_diffusion.hidden_dim, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim)
            )

    def forward(self, x, batch=None, return_embedding=False):
        output = self.graph_diffusion(x, batch)
        if return_embedding and self.use_contrastive:
            x_processed = self.graph_diffusion.lin1(x)
            if self.graph_diffusion.second_linear:
                x_processed = self.graph_diffusion.lin12(x_processed)
            graph_embed = self.graph_diffusion.graph_pooling(x_processed, batch)
            projected = self.projection_head(graph_embed)
            return output, projected
        return output

    def contrastive_loss(self, embeddings_list, labels_list):
        if len(embeddings_list) < 2:
            return torch.tensor(0.0, device=embeddings_list[0].device)
        all_embeddings = torch.cat(embeddings_list, dim=0)
        safe_labels = []
        for lbl in labels_list:
            if lbl.dim() == 0: lbl = lbl.unsqueeze(0)
            else: lbl = lbl.view(-1)
            safe_labels.append(lbl)
        all_labels = torch.cat(safe_labels, dim=0)
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        sim = torch.mm(all_embeddings, all_embeddings.t()) / self.temperature
        exp_sim = torch.exp(sim)
        labels = all_labels.unsqueeze(1)
        pos = (labels == labels.t()).float(); pos.fill_diagonal_(0)
        neg = 1 - pos; neg.fill_diagonal_(0)
        total, npos = 0, 0
        for i in range(len(all_labels)):
            pos_idx = torch.where(pos[i] > 0)[0]
            if len(pos_idx) == 0: continue
            neg_idx = torch.where(neg[i] > 0)[0]
            pos_sum = exp_sim[i, pos_idx].sum()
            neg_sum = exp_sim[i, neg_idx].sum()
            if pos_sum > 0 and (pos_sum + neg_sum) > 0:
                total = total - torch.log(pos_sum / (pos_sum + neg_sum))
                npos += 1
        return total / npos if npos > 0 else torch.tensor(0.0, device=all_embeddings.device)

    def compute_metrics_multiple(self, embeddings_list, data_list, split):
        if not isinstance(embeddings_list, list): embeddings_list = [embeddings_list]
        if not isinstance(data_list, list): data_list = [data_list]
        logits_list, labels_list, embeds_for_contrastive = [], [], []
        total_loss = 0
        for emb, data in zip(embeddings_list, data_list):
            if self.use_contrastive:
                logits, proj = self.forward(emb, return_embedding=True)
                embeds_for_contrastive.append(proj)
            else:
                logits = self.forward(emb)
            if 'graph_labels' in data: labels = data['graph_labels']
            elif 'y' in data: labels = data['y']
            elif 'labels' in data: labels = data['labels']
            else: raise KeyError("No graph labels found in data")
            labels = labels.view(-1).to(logits.device)
            logits_list.append(logits); labels_list.append(labels)
            if self.num_classes == 2:
                loss = self.criterion(logits.view(-1), labels.float())
            else:
                loss = self.criterion(logits, labels.long())
            total_loss += loss
        if self.use_contrastive and len(embeds_for_contrastive) > 1:
            cl = self.contrastive_loss(embeds_for_contrastive, labels_list)
        else:
            cl = torch.tensor(0.0, device=embeddings_list[0].device if len(embeddings_list)>0 else None)
        avg_cls = total_loss / max(1, len(logits_list))
        combined = avg_cls + self.contrastive_weight * cl if self.use_contrastive else avg_cls
        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        if self.num_classes == 2:
            probs = torch.sigmoid(all_logits.view(-1))
            preds = (probs > 0.5).float()
            labels_for_acc = all_labels.view(-1).float()
        else:
            probs = F.softmax(all_logits, dim=1)
            preds = torch.argmax(all_logits, dim=1)
            labels_for_acc = all_labels.view(-1).long()
        acc = (preds == labels_for_acc).float().mean()
        if self.num_classes == 2:
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                y = all_labels.view(-1).detach().cpu().numpy()
                p = probs.detach().cpu().numpy()
                roc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.0
                ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0.0
            except Exception as e:
                print(f"Warning: ROC/AP failed: {e}")
                roc, ap = 0.0, 0.0
        else:
            roc, ap = 0.0, 0.0
        return {
            'loss': combined,
            'classification_loss': avg_cls,
            'contrastive_loss': cl,
            'acc': acc.item(),
            'roc': roc,
            'ap': ap,
            'num_graphs': len(data_list)
        }

    def decode(self, x, batch=None):
        return self.forward(x, batch)

    def update_edge_index(self, edge_index):
        self.graph_diffusion.update_edge_index(edge_index)

    def grouped_parameters(self):
        return self.graph_diffusion.grouped_parameters()
