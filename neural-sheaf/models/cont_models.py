# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
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
    # def __init__(self,
    #              d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels,
    #              left_weights=False,
    #              right_weights=False,
    #              use_act=False,
    #              nonlinear=False,
    #              weight_learner=None):
    def __init__(self, edge_index, graph_size, d, beta=1.0, sigma=nn.ReLU(), c=1.0, diffusivity_learner=None):
        """
        Initialize the hyperbolic Laplacian ODE function.

        Args:
            edge_index (Tensor): Edge indices, shape (2, num_edges).
            graph_size (int): Number of nodes in the graph.
            d (int): Dimension of the hyperbolic space.
            beta (float): Parameter for distance-based diffusivity (default: 1.0).
            sigma (nn.Module): Activation function (default: nn.ReLU()).
            c (float): Curvature of the Poincaré ball (default: 1.0).
            diffusivity_learner (nn.Module, optional): Learned diffusivity module (default: None).
        """
        super(HyperbolicLaplacianODEFunc, self).__init__()
        self.edge_index = edge_index
        self.graph_size = graph_size
        self.d = d  # Dimension of hyperbolic space
        self.beta = beta  # Parameter for distance-based diffusivity
        self.sigma = sigma  # Activation function
        self.c = c  # Curvature of Poincaré ball
        self.diffusivity_learner = diffusivity_learner  # Optional learned diffusivity
        self.manifold = PoincareBall()  # Trustworthy Poincaré ball implementation
        if self.diffusivity_learner is None:
            self.norm_weights = self.compute_normalized_weights()
        else:
            self.norm_weights = None

    def compute_normalized_weights(self):
        
        deg = scatter_add(torch.ones_like(self.edge_index[0], dtype=torch.float), 
                        self.edge_index[0], dim=0, dim_size=self.graph_size)
        deg = deg.clamp(min=1e-6)
        source, target = self.edge_index
        deg_source = deg[source]
        deg_target = deg[target]
        norm_weights = 1 / (torch.sqrt(deg_source) * torch.sqrt(deg_target))
        return norm_weights

    def forward(self, t, x):
        """
        Compute the velocity (dx/dt) for the ODE solver.
        different from the paper, this func takes input and outputs the value in the tangent space.
        so in the solver, the value need to be projected back to the manifold (Poincaré ball).

        Args:
            t (float): Time point (scalar).
            x (Tensor): Node embeddings, shape (graph_size * d, 1).

        Returns:
            Tensor: Velocity dx/dt, shape (graph_size * d, 1).
        """
        # Reshape x to (graph_size, d)
        #print(f"forward start: x.shape = {x.shape}")
        x = x.view(self.graph_size, self.d)
        #print(f"reshaped x: {x.shape}")

        # Extract source and target embeddings based on edge_index
        source, target = self.edge_index
        z_source = x[source]  # Shape: (num_edges, d)
        z_target = x[target]  # Shape: (num_edges, d)
        #print(f"source.shape = {source.shape}, target.shape = {target.shape}")
        #print(f"z_source.shape = {z_source.shape}, z_target.shape = {z_target.shape}")
        # Compute logarithmic maps: log_z_i(z_j) for each edge (i, j)
        logmaps = self.manifold.logmap(z_source, z_target, c=self.c)  # Shape: (num_edges, d)
        #print(f"logmaps.shape = {logmaps.shape}")
        # Compute diffusivity a(z_i, z_j)
        if self.diffusivity_learner is not None:
            # Use learned diffusivity if provided
            a = self.diffusivity_learner(x, self.edge_index)  # Shape: (num_edges,)
        else:
            # # Default: Dirichlet Energy
            # sqdist = self.manifold.sqdist(z_source, z_target, c=self.c)  # Shape: (num_edges,)
            # a = torch.exp(-self.beta * sqdist)  # Shape: (num_edges,)
            a = self.norm_weights
        a = a.squeeze(-1)  # shape = (558,)
        #print(f"a.shape = {a.shape}")
        #print(f"a.unsqueeze(-1).shape = {a.unsqueeze(-1).shape}")
        # Weight logarithmic maps by diffusivity
        weighted_logmaps = a.unsqueeze(-1) * logmaps  # Shape: (num_edges, d)
        #print(f"weighted_logmaps.shape = {weighted_logmaps.shape}")
        # Aggregate over neighbors for each node i (sum over j where i is source)
        v = scatter_add(weighted_logmaps, source, dim=0, dim_size=self.graph_size)  # Shape: (graph_size, d)
        #print(v.shape)
        # Apply activation function
        #v = self.sigma(v)  # Shape: (graph_size, d), or directly use v if no activation is needed

        # exp_x^c(v)
        #v = self.manifold.expmap(v, x, c=self.c)  # Project back to manifold
        v=v.view(-1, 1)
        #print(f"forward output shape (velocity): {v.shape}")
        # Reshape velocity to match ODE solver input
        return v # Shape: (graph_size * d, 1)

'''class LaplacianODEFunc_(ODEFunc): # this implementation is from grand: neural graph diffusion

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    f = alpha * (ax - x)
    if self.opt['add_source']: #'If try get rid of alpha param and the beta*x0 source term')
      f = f + self.beta_train * self.x0
    return f
'''
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
        if self.opt["adjoint"] and self.training:
            print('111111111111111')
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
    """Diffusion model using hyperbolic Laplacian ODE."""

    def __init__(self, edge_index, args):
        super(HyperbolicGraphDiffusion, self).__init__(edge_index, args)
        # Hyperbolic dimension
        self.d = args.get('hyperbolic_dim', 2)  # Default to 2D hyperbolic space, adjust as needed
        self.hidden_dim = self.d  # Align hidden dimension with hyperbolic dimension
        #print("input_dim:", self.input_dim)
        # Input projection to hyperbolic space
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # Optional: Learn diffusivity similar to sheaf_learner
        self.diffusivity_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(1,), sheaf_act=self.sheaf_act) #if args.get('learn_diffusivity', False) else None

        # Define hyperbolic ODE function
        self.odefunc = HyperbolicLaplacianODEFunc(
            edge_index=edge_index,
            graph_size=self.graph_size,
            d=self.d,
            beta=args.get('beta', 1.0),
            sigma=nn.ReLU(),
            c=args.get('curvature', 1.0),
            diffusivity_learner=self.diffusivity_learner
        )
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)
        self.manifold = PoincareBall()
        self.sheaf_learners = [self.diffusivity_learner] if self.diffusivity_learner is not None else []
        if self.diffusivity_learner is not None:
            # 假设 learner 中定义了可学习参数 epsilons
            self.epsilons = list(self.diffusivity_learner.parameters())
        else:
            self.epsilons = []

    def forward(self, x):
        print("[DEBUG] lin1.weight nan?", torch.isnan(self.lin1.weight).any().item())
        print("[DEBUG] lin1.bias nan?", torch.isnan(self.lin1.bias).any().item())
        print("[DEBUG] lin1.weight max abs:", self.lin1.weight.abs().max().item())
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        print("[DEBUG] input x before lin1: nan?", torch.isnan(x).any().item())
        x = self.lin1(x)  # (graph_size, hidden_dim)
        print("[DEBUG] After lin1: nan?", torch.isnan(x).any().item())
        if self.use_act:
            x = F.elu(x)
        print("[DEBUG] After ELU: nan?", torch.isnan(x).any().item())
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        print("[DEBUG] After lin12: nan?", torch.isnan(x).any().item())
        # Project to Poincaré disk
        x = self.manifold.expmap0(x,1.0)  # Ensure x is in hyperbolic space
        print("[DEBUG] After expmap0: nan?", torch.isnan(x).any().item())
        if self.t > 0:
            x = x.view(self.graph_size * self.d, -1)  # (graph_size * d, 1)
            print("[DEBUG] Before odeblock: nan?", torch.isnan(x).any().item())
            x = self.odeblock(x)  # Apply hyperbolic diffusion
            print("[DEBUG] After odeblock: nan?", torch.isnan(x).any().item())
        x = self.manifold.logmap0(x,1.0)  # Project back to Poincaré disk
        print("[DEBUG] After logmap0: nan?", torch.isnan(x).any().item())
        x = x.view(self.graph_size, -1)  # (graph_size, d)
        x = self.lin2(x)  # Map to output dimension
        print("[DEBUG] After lin2: nan?", torch.isnan(x).any().item())
        out = F.log_softmax(x, dim=1)
        print("[DEBUG] After log_softmax: nan?", torch.isnan(out).any().item())
        return out

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


