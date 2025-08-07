import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException

from adjoint import odeint, odeint_adjoint # this lib is mini version of torchdiffeq 0.2.5
from hyperbolic_geometry import PoincareBall # # this code is part of the Hyperbolic Graph Convolutional Networks, source: https://github.com/HazyResearch/hgcn
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

  def forward(self, t, x, perturb=None):  # the t param is needed by the ODE solver.
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
    """Implements hyperbolic diffusion (Eq. 15) on the Poincaré ball."""
    def __init__(self, in_features, out_features, opt, data, device):
        super(HyperbolicGraphLaplacianODEFunc, self).__init__()
        # 从数据对象初始化图结构
        self.edge_index = data.edge_index.to(device)
        self.edge_weight = data.edge_attr
        if self.edge_weight is None:
            self.edge_weight = torch.ones(self.edge_index.size(1), device=device)
        else:
            self.edge_weight = self.edge_weight.to(device)

        self.graph_size = data.num_nodes
        # 使用隐藏维度作为超球维度
        self.d = opt.get('hidden_dim', in_features)
        self.device = device

        # 流形与超曲率
        self.c = float(opt.get('curvature', 1.0))
        self.manifold = PoincareBall()

        act = opt.get('hyperbolic_activation', 'identity')
        if act == 'tanh':
            self.sigma = torch.tanh
        elif act == 'relu':
            self.sigma = torch.relu
        else:
            self.sigma = lambda x: x

        # 计数器（与其它 ODEFunc 对齐的接口字段，便于上层统计 NFE）
        self.nfe = 0
        self.opt = opt
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 新增：速度缩放，初始较小

    def forward(self, t, x, perturb=None):
        """
        输入与输出均在切空间中进行演化，外部求解器负责在步进间做投影/映射。
        x: (N, d)
        """
        self.nfe += 1

        # 保证形状 (N, d)
        if x.dim() == 2 and x.size(0) == self.graph_size:
            z = x
        else:
            z = x.view(self.graph_size, -1)

        # 取边两端嵌入
        src, dst = self.edge_index[0], self.edge_index[1]
        z_src = z[src]  # (E, d)
        z_dst = z[dst]  # (E, d)

        # 在超曲空间计算对数映射：log_{z_src}(z_dst)
        logmaps = self.manifold.logmap(z_src, z_dst, c=self.c)  # (E, d)
        logmaps = torch.clamp(logmaps, min=-10.0, max=10.0)

        # 权重
        ew = self.edge_weight.unsqueeze(-1)  # (E, 1)
        ew = torch.clamp(ew, min=0.0, max=10.0)
        weighted = ew * logmaps  # (E, d)

        # 聚合到源节点
        v = scatter_add(weighted, src, dim=0, dim_size=self.graph_size)  # (N, d)

        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        v = self.sigma(v)
        v = self.gamma * v  # 可学缩放，防止速度过大/过小
        return v
