# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, LocalConcatHodgeSheafLearner
from lib.adjoint import odeint, odeint_adjoint
from lib.hyperbolic_geometry import PoincareBall
from utils import MaxNFEException
import torch_sparse


class MaxNFEException(Exception):
    """Exception raised when maximum number of function evaluations is exceeded"""
    pass


class HyperbolicLaplacianODEFunc(nn.Module):
    """Implements hyperbolic diffusion based on Equation 15 using PoincareBall."""
    def __init__(self, 
                d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels, 
                left_weights=False,
                right_weights=False,
                use_act=False,
                nonlinear=False,
                weight_learner=None,
                edge_attr=None,
                max_nfe=1000):

        super(HyperbolicLaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner  # Optional learned diffusivity
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.edge_attr = edge_attr if edge_attr is not None else torch.ones(edge_index.size(1), 1)
        self.nonlinear = nonlinear
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        
        self.L = None
        self.nfe = 0  # Number of function evaluations
        self.max_nfe = max_nfe
        self.sigma = nn.ReLU()  # Activation function
        
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

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

        # Check maximum number of function evaluations
        if self.nfe > self.max_nfe:
            raise MaxNFEException
        self.nfe += 1

        # 保证形状为 (graph_size, final_d * hidden_channels)
        if x.dim() == 2 and x.size(0) == self.graph_size:
            z = x
        else:
            z = x.view(self.graph_size, -1)

        # 取边两端嵌入
        src, dst = self.edge_index[0], self.edge_index[1]
        z_src = z[src]  # (E, final_d * hidden_channels)
        z_dst = z[dst]  # (E, final_d * hidden_channels)

        # 在超曲空间计算对数映射：log_{z_src}(z_dst)
        logmaps = self.manifold.logmap(z_src, z_dst, c=self.c)  # (E, final_d * hidden_channels)
        
        if torch.isnan(logmaps).any() or torch.isinf(logmaps).any():
            print("Warning: NaN or Inf detected in logmaps")
            logmaps = torch.where(torch.isfinite(logmaps), logmaps, torch.zeros_like(logmaps))
        
        logmaps = torch.clamp(logmaps, min=-5.0, max=5.0)  # 更严格的clamp

        if self.edge_attr is not None:
            ew = self.edge_attr.to(logmaps.device)
        else:
            ew = torch.ones(logmaps.size(0), 1, device=logmaps.device)
        ew = torch.clamp(ew, min=0.0, max=10.0)
        weighted = ew * logmaps  # (E, final_d * hidden_channels)

        # 聚合到源节点
        v = scatter_add(weighted, src, dim=0, dim_size=self.graph_size)  # (N, final_d * hidden_channels)

        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))

        if self.use_act:
            # x = F.elu(x)
            v = self.sigma(v) 

        # v = self.manifold.proj_tan(v, c=self.c)

        v = self.manifold.expmap(z, v, c=self.c)  
        v = self.manifold.proj(v, c=self.c)  

        # 从 (graph_size, final_d * hidden_channels) 转回 (graph_size * final_d, hidden_channels)
        v = v.reshape(self.graph_size * self.d, -1)
        
        return v  # Shape: (graph_size * final_d, hidden_channels)


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
        if self.opt["adjoint"] and self.training:
            z = odeint_adjoint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'], # set the numerical solver: hyperbolic_rk4, rk4, dopri5
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters'], manifold=self.manifold),
                adjoint_method=self.opt['adjoint_method'], # "set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
                adjoint_options=dict(step_size=self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint)
        else:
            #print('222222222222')
            z = odeint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters'],  manifold=self.manifold),
                atol=self.atol,
                rtol=self.rtol)
        self.odefunc.L = None
        z = z[1]
        return z


class HyperbolicGraphDiffusion(SheafDiffusion):
    """Diffusion model using hyperbolic Laplacian ODE for brain network classification."""

    def __init__(self, edge_index, args):
        super(HyperbolicGraphDiffusion, self).__init__(edge_index, args)
        # Hyperbolic dimension
        # self.d = args.get('hyperbolic_dim', 2)  # defined in the SheafDiffusion class
        self.c = 1.0 # hyperbolic curvature
        
        # Store edge attributes (brain network weights)
        self.edge_attr = None
        
        # Store original edge_index (single graph structure)
        self.edge_index = edge_index
        
        # Graph pooling method
        self.pool_type = args.get('pool_type', 'mean')
        
        # Input projection to hyperbolic space
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Graph-level output projection
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # Initialize manifold
        self.manifold = PoincareBall()

        # Initialize sheaf learner with correct parameters: (hidden_dim, d)
        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, (self.d,))
        
        # Initialize Laplacian builder
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)


        # Define hyperbolic ODE function
        self.odefunc = HyperbolicLaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, edge_attr=self.edge_attr, max_nfe=args.get('max_nfe', 1000))
        
        # Initialize ODE block with complete options
        ode_opt = {
            'int_method': args.get('int_method', 'hyperbolic_rk4'),
            'step_size': args.get('step_size', 0.5),
            'max_iters': args.get('max_nfe', 1000),
            'adjoint': args.get('adjoint', False),
            'adjoint_method': args.get('adjoint_method', 'rk4'),
            'adjoint_step_size': args.get('adjoint_step_size', 1.0),
            'tol_scale': args.get('tol_scale', 1.0),
            'tol_scale_adjoint': args.get('tol_scale_adjoint', 1.0)
        }
        self.odeblock = ODEBlock(self.odefunc, self.time_range, ode_opt)


    def forward(self, data):
        """
        Forward pass for brain network classification.
        
        Args:
            data: PyG Data object with:
                - x: [batch_size * n_nodes, n_nodes] or batched format
                - batch: [batch_size * n_nodes] batch assignment
                - y: [batch_size] labels
        
        Returns:
            out: [batch_size, output_dim] log probabilities
        """
        # Reset function evaluation counter
        self.odefunc.nfe = 0
        
        if hasattr(data, 'batch'):
            batch_size = data.batch.max().item() + 1
        else:
            batch_size = 1
        
        x = data.x  # (batch_size * graph_size, input_dim)
        
        # 投影到隐藏维度
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)  # (batch_size * graph_size, hidden_dim)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        # 投影到Poincaré圆盘
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        
        # Reshape: (batch_size * graph_size, hidden_dim) -> (batch_size, graph_size, hidden_dim)
        x = x.view(batch_size, self.graph_size, -1)

        if self.t > 0:
            x_out = []
            
            edge_index_device = self.edge_index.to(x.device)
            self.odefunc.edge_index = edge_index_device
            self.sheaf_learner.edge_index = edge_index_device
            
            if not hasattr(self, '_laplacian_device') or self._laplacian_device != x.device:
                self.odefunc.laplacian_builder = lb.DiagLaplacianBuilder(
                    self.graph_size, edge_index_device, d=self.d, 
                    add_hp=self.add_hp, add_lp=self.add_lp)
                self._laplacian_device = x.device
            
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                self.odefunc.edge_attr = data.edge_attr[:self.edge_index.size(1)]
            else:
                self.odefunc.edge_attr = torch.ones(self.edge_index.size(1), 1, device=x.device)
            
            for i in range(batch_size):
                x_single = x[i]  # (graph_size, hidden_dim)
                x_single = x_single.view(self.graph_size * self.final_d, -1)
                x_single = self.odeblock(x_single)
                x_single = x_single.view(self.graph_size, -1)
                x_out.append(x_single)
            
            x = torch.stack(x_out, dim=0)  # (batch_size, graph_size, hidden_dim)

        # 投影回切空间
        x = self.manifold.logmap0(x.view(-1, x.size(-1)), self.c)
        x = x.view(batch_size, self.graph_size, -1)
        
        # pooling
        if self.pool_type == 'mean':
            x = x.mean(dim=1)
        elif self.pool_type == 'max':
            x = x.max(dim=1)[0]
        elif self.pool_type == 'sum':
            x = x.sum(dim=1)
        elif self.pool_type == 'attention':
            attn_weights = torch.softmax(x.mean(dim=-1), dim=-1).unsqueeze(-1)
            x = (x * attn_weights).sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # 分类
        x = self.lin2(x)
        out = F.log_softmax(x, dim=1)

        return out

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.edge_index = edge_index
        # diffusivity_learner在这个模型中是sheaf_learner
        if hasattr(self, 'sheaf_learner') and self.sheaf_learner is not None:
            self.sheaf_learner.edge_index = edge_index




# ------------- Hyperbolic Hodge Laplacian ODE Function ------------- #

class HyperbolicHodgeLaplacianODEFunc(nn.Module):
    def __init__(self, 
                d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels, 
                left_weights=False,
                right_weights=False,
                use_act=False,
                nonlinear=False,
                weight_learner=None,
                edge_attr=None,
                max_nfe=1000):

        super(HyperbolicHodgeLaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner  # Optional learned diffusivity
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.edge_attr = edge_attr if edge_attr is not None else torch.ones(edge_index.size(1), 1)
        self.nonlinear = nonlinear
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        
        self.L = None
        self.nfe = 0  # Number of function evaluations
        self.max_nfe = max_nfe
        self.sigma = nn.ReLU()  # Activation function
        
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

        self.c = 1.0  # Curvature of Poincaré ball
        self.manifold = PoincareBall()  # Trustworthy Poincaré ball implementation

    def forward(self, t, x):
        """
        x : edge features, shape ≈ (num_edges * d, hidden_channels) or (num_edges, d*hidden_channels)
        """
        # HyperbolicLaplacianODEFunc 里跳过了sheaflearner， 独立实现了固定的图拉普拉斯算子
        # 这里使用 sheaf_learner 来学习 Hodge Laplacian： L
        # 相关代码基于原版LaplacianODEFunc修改
        self.nfe += 1
        x = self.manifold.logmap0(x, c=self.c)

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
            x = self.sigma(x)

        # Riemannian transport back
        v_tan = self.manifold.proj_tan0(x, c=self.c)
        new_z = self.manifold.expmap0(v_tan, c=self.c)
        new_z = self.manifold.proj(new_z, c=self.c)

        # Reshape back to original format
        return new_z.view(-1, self.hidden_channels)




class HyperbolicSimplicialDiffusion(SheafDiffusion):
    """Diffusion model using hyperbolic Laplacian ODE for brain network classification."""
    def __init__(self, edge_index, args):
        super(HyperbolicSimplicialDiffusion, self).__init__(edge_index, args)
        # Hyperbolic dimension
        # self.d = args.get('hyperbolic_dim', 2)  # defined in the SheafDiffusion class
        self.c = 1.0 # hyperbolic curvature
        
        # Store edge attributes (brain network weights)
        self.edge_attr = None
        
        # Store original edge_index (single graph structure)
        self.edge_index = edge_index
        
        # Graph pooling method
        self.pool_type = args.get('pool_type', 'mean')
        
        # Input projection to hyperbolic space
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Graph-level output projection
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # Initialize manifold
        self.manifold = PoincareBall()

        # Initialize sheaf learner with correct parameters: (hidden_dim, d)
        self.sheaf_learner = LocalConcatHodgeSheafLearner(self.hidden_dim, (self.d,))
        
        # Initialize Laplacian builder
        self.laplacian_builder = lb.DiagHodgeLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)


        # Define hyperbolic ODE function
        self.odefunc = HyperbolicHodgeLaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, edge_attr=self.edge_attr, max_nfe=args.get('max_nfe', 1000))
        
        # Initialize ODE block with complete options
        ode_opt = {
            'int_method': args.get('int_method', 'hyperbolic_rk4'),
            'step_size': args.get('step_size', 0.5),
            'max_iters': args.get('max_nfe', 1000),
            'adjoint': args.get('adjoint', False),
            'adjoint_method': args.get('adjoint_method', 'rk4'),
            'adjoint_step_size': args.get('adjoint_step_size', 1.0),
            'tol_scale': args.get('tol_scale', 1.0),
            'tol_scale_adjoint': args.get('tol_scale_adjoint', 1.0)
        }
        self.odeblock = ODEBlock(self.odefunc, self.time_range, ode_opt)


    def forward(self, data):
        """
        Forward pass for brain network classification.
        
        Args:
            data: PyG Data object with:
                - x: [batch_size * n_nodes, n_nodes] or batched format
                - batch: [batch_size * n_nodes] batch assignment
                - y: [batch_size] labels
        
        Returns:
            out: [batch_size, output_dim] log probabilities
        """
        # Reset function evaluation counter
        self.odefunc.nfe = 0
        
        if hasattr(data, 'batch'):
            batch_size = data.batch.max().item() + 1
        else:
            batch_size = 1
        
        x = data.x  # (batch_size * graph_size, input_dim)
        
        # 投影到隐藏维度
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)  # (batch_size * graph_size, hidden_dim)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        # 投影到Poincaré圆盘
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        
        # Reshape: (batch_size * graph_size, hidden_dim) -> (batch_size, graph_size, hidden_dim)
        x = x.view(batch_size, self.graph_size, -1)

        if self.t > 0:
            x_out = []
            
            edge_index_device = self.edge_index.to(x.device)
            self.odefunc.edge_index = edge_index_device
            self.sheaf_learner.edge_index = edge_index_device
            
            if not hasattr(self, '_laplacian_device') or self._laplacian_device != x.device:
                self.odefunc.laplacian_builder = lb.DiagLaplacianBuilder(
                    self.graph_size, edge_index_device, d=self.d, 
                    add_hp=self.add_hp, add_lp=self.add_lp)
                self._laplacian_device = x.device
            
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                self.odefunc.edge_attr = data.edge_attr[:self.edge_index.size(1)]
            else:
                self.odefunc.edge_attr = torch.ones(self.edge_index.size(1), 1, device=x.device)
            
            for i in range(batch_size):
                x_single = x[i]  # (graph_size, hidden_dim)
                x_single = x_single.view(self.graph_size * self.final_d, -1)
                x_single = self.odeblock(x_single)
                x_single = x_single.view(self.graph_size, -1)
                x_out.append(x_single)
            
            x = torch.stack(x_out, dim=0)  # (batch_size, graph_size, hidden_dim)

        # 投影回切空间
        x = self.manifold.logmap0(x.view(-1, x.size(-1)), self.c)
        x = x.view(batch_size, self.graph_size, -1)
        
        # pooling
        if self.pool_type == 'mean':
            x = x.mean(dim=1)
        elif self.pool_type == 'max':
            x = x.max(dim=1)[0]
        elif self.pool_type == 'sum':
            x = x.sum(dim=1)
        elif self.pool_type == 'attention':
            attn_weights = torch.softmax(x.mean(dim=-1), dim=-1).unsqueeze(-1)
            x = (x * attn_weights).sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # 分类
        x = self.lin2(x)
        out = F.log_softmax(x, dim=1)

        return out

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.edge_index = edge_index
        # diffusivity_learner在这个模型中是sheaf_learner
        if hasattr(self, 'sheaf_learner') and self.sheaf_learner is not None:
            self.sheaf_learner.edge_index = edge_index





            