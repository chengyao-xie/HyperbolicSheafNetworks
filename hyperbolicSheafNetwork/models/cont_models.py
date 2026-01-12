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
    def __init__(self, d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels,
                 left_weights=False, right_weights=False, use_act=False, nonlinear=False, weight_learner=None):

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
        self.gamma = nn.Parameter(torch.tensor(0.1)) 
        
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
            x_maps = x.view(self.graph_size, -1)
            maps = self.sheaf_learner(x_maps, self.edge_index)
            if self.weight_learner is not None:
                self.edge_weights = self.weight_learner(x_maps, self.edge_index)
                L, _ = self.laplacian_builder(maps, self.edge_weights)
            else:
                L, _ = self.laplacian_builder(maps)
                self.edge_weights = torch.ones(self.edge_index.size(1), 1, device=x.device)
            self.L = L
        else:
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

        if self.edge_weights is None:
            self.edge_weights = torch.ones(self.edge_index.size(1), 1, device=x.device)
        
        ew = self.edge_weights
        if ew.dim() == 1:
            ew = ew.unsqueeze(-1)
        ew = torch.clamp(ew, min=0.0, max=10.0)
        
        # ew = self.edge_weights.unsqueeze(-1)  # (E, 1)
        # weighted = ew * logmaps  # (E, d)
        # v = scatter_add(weighted, src, dim=0, dim_size=self.graph_size)
        
        # 计算节点度数
        deg_src = scatter_add(torch.ones_like(src, dtype=torch.float), src, dim=0, dim_size=self.graph_size)
        deg_dst = scatter_add(torch.ones_like(dst, dtype=torch.float), dst, dim=0, dim_size=self.graph_size)
        
        # 对称归一化: w_ij / sqrt(deg_i * deg_j)
        norm_factor = torch.sqrt(deg_src[src] * deg_dst[dst] + 1e-8)
        ew_normalized = ew.squeeze(-1) / norm_factor
        ew_normalized = ew_normalized.unsqueeze(-1)
        
        weighted = ew_normalized * logmaps  # (E, d)
        v = scatter_add(weighted, src, dim=0, dim_size=self.graph_size)  # (N, d)

        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))

        # v = torch.clamp(self.gamma, min=0.001, max=0.5) * v
        
        gamma_positive = F.softplus(self.gamma)
        v = gamma_positive * v
        
        # v = self.manifold.expmap(z, v, c=self.c)
        
        v = self.manifold.proj_tan(v, z, c=self.c)  # 确保在切空间
        v_clipped = torch.clamp(v, min=-1.0, max=1.0)  # 限制切向量范数
        v = self.manifold.expmap(z, v_clipped, c=self.c)
        
        # Ensure output is on the hyperbolic disk
        v = self.manifold.proj(v, c=self.c)

        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        v = torch.clamp(v, min=-5, max=5)
        
        # Ensure output shape matches input shape
        v = v.view(-1)  # Reshape to (graph_size * d,)

        
        return v # Shape: (graph_size * d,)


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
        self.c = 1.0
        print("input_dim:", self.input_dim)
        
        if not hasattr(self, 'int_method'):
            self.int_method = 'hyperbolic_rk4'
    
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
            
        self.pool_type = args.get('pool_type', 'mean')
        
        if self.pool_type == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 2, 1)
            )
        
        self.graph_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),  
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, (self.d,))
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)

        self.odefunc = HyperbolicLaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)
        self.manifold = PoincareBall()

        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.residual = args.get('residual', True)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        self._cached_graph_size = self.graph_size
        self._cached_edge_index_hash = self._hash_edge_index(edge_index)
        
        self.reset_parameters()
    
    def _hash_edge_index(self, edge_index):
        return (edge_index.shape[1], edge_index.sum().item())
        
    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)
        
        # 分类器最后一层特殊初始化，使用init_bias
        last_layer = self.graph_classifier[-1]
        if isinstance(last_layer, nn.Linear):
            # 使用更合理的权重初始化
            nn.init.xavier_uniform_(last_layer.weight, gain=0.1)
            if last_layer.bias is not None:
                init_bias = getattr(self, '_init_bias', 0.0)
                nn.init.constant_(last_layer.bias, init_bias)
                print(f"[INFO] Initialized classifier bias to {init_bias:.4f}")
        
        # gamma参数初始化
        with torch.no_grad():
            if hasattr(self.odefunc, 'gamma'):
                self.odefunc.gamma.fill_(0.1)

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
                
    def update_graph(self, edge_index, graph_size):
        """更新图结构"""
        self.graph_size = graph_size
        self.edge_index = edge_index
        
        # 重新创建 laplacian_builder
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)
        
        # 更新 odefunc
        self.odefunc.graph_size = graph_size
        self.odefunc.edge_index = edge_index
        self.odefunc.laplacian_builder = self.laplacian_builder
        self.odefunc.L = None  # 重置缓存的拉普拉斯矩阵
                
    def forward(self, data_or_x, batch=None):
        """前向传播"""
        if hasattr(data_or_x, 'x'):
            x = data_or_x.x
            edge_index = data_or_x.edge_index
            graph_size = x.size(0)
            
            # 使用哈希快速检查是否需要更新
            new_hash = self._hash_edge_index(edge_index)
            if graph_size != self._cached_graph_size or new_hash != self._cached_edge_index_hash:
                self.update_graph(edge_index, graph_size)
                self._cached_graph_size = graph_size
                self._cached_edge_index_hash = new_hash
        else:
            x = data_or_x
        
        # Input processing
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        
        # First transformation layer
        x = self.lin1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 可选的第二个线性层
        if self.second_linear:
            x = self.lin12(x)
            x = self.layer_norm2(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # 投影到双曲空间
        x_scaled = torch.tanh(x) * 0.9  # 从0.8增加到0.9
        x = self.manifold.expmap0(x_scaled, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        
        # ODE 演化 
        if self.t > 0 and self.training:
            if torch.rand(1).item() < 0.8: 
                x_flat = x.view(-1)
                if torch.isnan(x_flat).any() or torch.isinf(x_flat).any():
                    x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=1.0, neginf=-1.0)
                
                x_evolved = self.odeblock(x_flat)
                
                if torch.isnan(x_evolved).any() or torch.isinf(x_evolved).any():
                    x_evolved = torch.nan_to_num(x_evolved, nan=0.0, posinf=5.0, neginf=-5.0)
                
                x = x_evolved.view(self.graph_size, -1)
        elif self.t > 0:  
            x_flat = x.view(-1)
            if torch.isnan(x_flat).any() or torch.isinf(x_flat).any():
                x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=1.0, neginf=-1.0)
            
            x_evolved = self.odeblock(x_flat)
            
            if torch.isnan(x_evolved).any() or torch.isinf(x_evolved).any():
                x_evolved = torch.nan_to_num(x_evolved, nan=0.0, posinf=5.0, neginf=-5.0)
            
            x = x_evolved.view(self.graph_size, -1)
        
        # Project back to Euclidean space
        x = self.manifold.logmap0(x, self.c)
        x = torch.clamp(x, min=-10, max=10)
        
        # 图分类输出
        graph_embed = self.graph_pooling(x, batch)
        logit = self.graph_classifier(graph_embed)
        
        if torch.isnan(logit).any() or torch.isinf(logit).any():
            logit = torch.nan_to_num(logit, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return logit


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
        
        # 传递初始偏置到模型
        init_bias = args_copy.get('init_bias', 0.0)
        
        # 创建超曲图扩散模型
        self.graph_diffusion = HyperbolicGraphDiffusion(edge_index, args_copy)
        
        # 保存初始偏置供reset_parameters使用
        self.graph_diffusion._init_bias = init_bias
        
        # 使用Focal Loss替代BCEWithLogitsLoss
        self.num_classes = args_copy['num_classes']
        if self.num_classes == 2:
            focal_alpha = args_copy.get('focal_alpha', 0.75)
            focal_gamma = args_copy.get('focal_gamma', 2.0)
            
            # 导入Focal Loss（需要在文件开头定义）
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                
                def forward(self, inputs, targets):
                    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                    pt = torch.exp(-bce_loss)
                    focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                    return focal_loss.mean()
            
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"[INFO] Using Focal Loss with alpha={focal_alpha:.4f}, gamma={focal_gamma:.4f}, init_bias={init_bias:.4f}")
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, batch=None):
        """前向传播"""
        return self.graph_diffusion(x, batch)
    
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






# class EuclideanGraphDiffusion(SheafDiffusion):
#     """欧氏空间图扩散（消融：用欧氏 ODE 替换双曲 ODE）"""

#     def __init__(self, edge_index, args):
#         super(EuclideanGraphDiffusion, self).__init__(edge_index, args)
#         if isinstance(args, dict):
#             args.setdefault('int_method', 'rk4')
#         else:
#             if not hasattr(args, 'int_method'):
#                 setattr(args, 'int_method', 'rk4')

#         self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
#         if self.second_linear:
#             self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

#         self.pool_type = args.get('pool_type', 'mean')
#         if self.pool_type == 'attention':
#             self.attention_pool = nn.Sequential(
#                 nn.Linear(self.hidden_dim, self.hidden_dim // 2),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_dim // 2, 1)
#             )
#         self.graph_classifier = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(self.hidden_dim // 2, self.output_dim)
#         )

#         self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, (self.d,), sheaf_act=self.sheaf_act)
#         self.laplacian_builder = lb.DiagLaplacianBuilder(
#             self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)

#         self.odefunc = HyperbolicLaplacianODEFunc(
#             self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index,
#             self.graph_size, self.hidden_channels,
#             nonlinear=self.nonlinear, left_weights=self.left_weights,
#             right_weights=self.right_weights, use_act=self.use_act)
#         self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

#         self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
#         self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
#         self.residual = args.get('residual', True)
#         self.activation = nn.LeakyReLU(negative_slope=0.2)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.lin1.weight)
#         for layer in self.graph_classifier:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#         if self.second_linear:
#             nn.init.xavier_uniform_(self.lin12.weight)

#     def graph_pooling(self, x, batch=None):
#         if self.pool_type == 'mean':
#             if batch is None:
#                 return torch.mean(x, dim=0, keepdim=True)
#             else:
#                 from torch_geometric.nn import global_mean_pool
#                 return global_mean_pool(x, batch)
#         elif self.pool_type == 'max':
#             if batch is None:
#                 return torch.max(x, dim=0, keepdim=True)[0]
#             else:
#                 from torch_geometric.nn import global_max_pool
#                 return global_max_pool(x, batch)
#         elif self.pool_type == 'sum':
#             if batch is None:
#                 return torch.sum(x, dim=0, keepdim=True)
#             else:
#                 from torch_geometric.nn import global_add_pool
#                 return global_add_pool(x, batch)
#         elif self.pool_type == 'attention':
#             att = self.attention_pool(x)
#             att = F.softmax(att, dim=0)
#             if batch is None:
#                 return torch.sum(x * att, dim=0, keepdim=True)
#             else:
#                 graph_embed = []
#                 for i in range(batch.max().item() + 1):
#                     mask = (batch == i)
#                     xg = x[mask]
#                     attg = F.softmax(att[mask], dim=0)
#                     graph_embed.append(torch.sum(xg * attg, dim=0))
#                 return torch.stack(graph_embed)
#         else:
#             raise ValueError(f"Unknown pooling type: {self.pool_type}")

#     def update_graph(self, edge_index, graph_size):
#         """更新图结构"""
#         self.graph_size = graph_size
#         self.edge_index = edge_index
        
#         # 重新创建 laplacian_builder
#         self.laplacian_builder = lb.DiagLaplacianBuilder(
#             graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)
        
#         # 更新 odefunc
#         self.odefunc.graph_size = graph_size
#         self.odefunc.edge_index = edge_index
#         self.odefunc.laplacian_builder = self.laplacian_builder
#         self.odefunc.L = None

#     def forward(self, x, batch=None):
#         x = F.dropout(x, p=self.input_dropout, training=self.training)
#         x = self.lin1(x)
#         x = self.layer_norm1(x)
#         x = self.activation(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         if self.second_linear:
#             x = self.lin12(x)
#             x = self.layer_norm2(x)
#             x = self.activation(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         if self.t > 0:
#             x = x.view(self.graph_size * self.final_d, -1)
#             x = self.odeblock(x)
#         x = x.view(self.graph_size, -1)

#         graph_embed = self.graph_pooling(x, batch)
#         output = self.graph_classifier(graph_embed)
#         if self.output_dim == 2:
#             return output[:, 0:1]
#         else:
#             return F.log_softmax(output, dim=1)

#     def update_edge_index(self, edge_index):
#         super().update_edge_index(edge_index)
#         self.odefunc.update_laplacian_builder(self.laplacian_builder)


# class EuclideanGraphClassifier(nn.Module):
#     """欧氏空间图分类器（与 HyperbolicGraphClassifier 同接口，便于消融对比）"""
#     def __init__(self, edge_index, args):
#         super(EuclideanGraphClassifier, self).__init__()
#         self.args = args
#         self.task = 'graph_classification'

#         if isinstance(args, dict):
#             args_copy = args.copy()
#         else:
#             args_copy = vars(args).copy()
#         args_copy['graph_classification'] = True
#         args_copy['num_classes'] = args_copy.get('num_classes', 2)
#         args_copy.setdefault('int_method', 'rk4')

#         self.graph_diffusion = EuclideanGraphDiffusion(edge_index, args_copy)

#         self.num_classes = args_copy['num_classes']
#         self.criterion = nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()

#     def forward(self, x, batch=None):
#         """前向传播"""
#         return self.graph_diffusion(x, batch)
