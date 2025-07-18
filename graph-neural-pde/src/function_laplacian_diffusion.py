import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException

from lib.adjoint import odeint, odeint_adjoint # this lib is mini version of torchdiffeq 0.2.5
from lib.hyperbolic_geometry import PoincareBall # # this code is part of the Hyperbolic Graph Convolutional Networks, source: https://github.com/HazyResearch/hgcn
import torch_sparse
from torch_scatter import scatter_add

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

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




class HyperbolicGraphLaplacianODEFunc(nn.Module):
    """Implements hyperbolic diffusion based on Equation 15 using PoincareBall."""
    def __init__(self, in_features, out_features, opt, data, device):

        super(HyperbolicGraphLaplacianODEFunc, self).__init__()
        self.edge_index = edge_index  # edge_index is processed in neural-sheaf/utils/heterophilic.py get_dataset() function
        self.graph_size = graph_size
        self.d = d  # defined in neural-sheaf, hidden_dim, default is 2
        self.c = 1.0  # Curvature of Poincaré ball
        self.manifold = PoincareBall()  # Trustworthy Poincaré ball implementation



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
        x = x.view(self.graph_size, self.d)

        # Extract source and target embeddings based on edge_index
        source, target = self.edge_index
        z_source = x[source]  # Shape: (num_edges, d)
        z_target = x[target]  # Shape: (num_edges, d)

        # Compute logarithmic maps: log_z_i(z_j) for each edge (i, j)
        logmaps = self.manifold.logmap(z_source, z_target, c=self.c)  # Shape: (num_edges, d)

        edge_weight_matrix = self.edge_weight

        # Weight logarithmic maps by diffusivity
        weighted_logmaps = edge_weight_matrix.unsqueeze(-1) * logmaps  # Shape: (num_edges, d)

        # Aggregate over neighbors for each node i (sum over j where i is source)
        v = scatter_add(weighted_logmaps, source, dim=0, dim_size=self.graph_size)  # Shape: (graph_size, d)

        # Apply activation function
        v = self.sigma(v)  # Shape: (graph_size, d), or directly use v if no activation is needed

        # exp_x^c(v)
        v = self.manifold.expmap(v, x, c=self.c)  # Project back to manifold

        # Reshape velocity to match ODE solver input
        return v.view(-1, 1)  # Shape: (graph_size * d, 1)
