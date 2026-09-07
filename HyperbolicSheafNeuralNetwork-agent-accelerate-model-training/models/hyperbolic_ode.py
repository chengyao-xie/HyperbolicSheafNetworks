import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn
from torch_scatter import scatter_add

from lib.adjoint import odeint, odeint_adjoint
from lib.hyperbolic_geometry import PoincareBall


class MaxNFEException(Exception):
    """Exception raised when maximum number of function evaluations is exceeded."""
    pass


class HyperbolicLaplacianODEFunc(nn.Module):
    """Hyperbolic sheaf diffusion ODE with optional fixed Hodge correction."""

    def __init__(self,
                 d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels,
                 left_weights=False,
                 right_weights=False,
                 use_act=False,
                 weight_learner=None,
                 edge_attr=None,
                 max_nfe=1000,
                 curvature=1.0,
                 learn_curvature=False,
                 simplex_order=2,
                 use_learned_sheaf=True,
                 use_fixed_hodge=True,
                 hodge_weight=1.0):
        super(HyperbolicLaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.edge_attr = edge_attr if edge_attr is not None else torch.ones(edge_index.size(1), 1)
        self.B1 = None
        self.hodge_laplacian = None
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        self.simplex_order = simplex_order
        self.use_learned_sheaf = use_learned_sheaf
        self.use_fixed_hodge = use_fixed_hodge
        self.hodge_weight = hodge_weight

        self.L = None
        self.nfe = 0
        self.max_nfe = max_nfe
        self.sigma = nn.ReLU()

        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

        if learn_curvature:
            self.c_param = nn.Parameter(torch.tensor(curvature, dtype=torch.float32))
            self.learn_curvature = True
        else:
            self.register_buffer('c_param', torch.tensor(curvature, dtype=torch.float32))
            self.learn_curvature = False
        self.manifold = PoincareBall()

    # 更新当前样本对应的图结构和边界矩阵。
    def set_graph_context(self, edge_index, edge_attr=None, B1=None, hodge_laplacian=None,
                          laplacian_builder=None, graph_size=None):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.B1 = B1
        self.hodge_laplacian = hodge_laplacian
        if graph_size is not None:
            self.graph_size = graph_size
        if laplacian_builder is not None:
            self.laplacian_builder = laplacian_builder

    def get_curvature(self):
        return F.softplus(self.c_param)

    # 执行 ODE 前向向量场计算。
    def forward(self, t, x):
        self._check_nfe()
        curvature = self.get_curvature()
        state = self._as_state(x)
        state_tan = self.manifold.logmap0(state, c=curvature)
        state_tan = self._apply_channel_weights(state_tan)
        node_tan = state_tan.view(self.graph_size, -1)

        tangent = self._learned_sheaf_tangent(state_tan, node_tan)
        if tangent is None:
            tangent = self._edge_tangent(node_tan).view_as(state_tan)

        if self.simplex_order == 2 and self.use_fixed_hodge:
            hodge_tangent = self._fixed_hodge_tangent(node_tan).view_as(state_tan)
            tangent = tangent + self.hodge_weight * hodge_tangent

        tangent = torch.where(torch.isfinite(tangent), tangent, torch.zeros_like(tangent))
        if self.use_act:
            tangent = self.sigma(tangent)

        next_state = self.manifold.expmap(tangent, state, c=curvature)
        next_state = self.manifold.proj(next_state, c=curvature)
        return next_state

    # 检查 ODE 函数调用次数。
    def _check_nfe(self):
        if self.nfe > self.max_nfe:
            raise MaxNFEException
        self.nfe += 1

    # 整理 ODE 状态形状。
    def _as_state(self, x):
        if x.dim() == 2 and x.size(0) == self.graph_size * self.d:
            return x
        if x.dim() == 2 and x.size(0) == self.graph_size:
            return x.view(self.graph_size * self.d, -1)
        return x.view(self.graph_size * self.d, -1)

    # 应用左右线性权重。
    def _apply_channel_weights(self, state_tan):
        x = state_tan
        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, self.graph_size * self.d).t()
        if self.right_weights:
            x = self.lin_right_weights(x)
        return x

    # 用 learnable sheaf restriction maps 构造扩散切向量。
    def _learned_sheaf_tangent(self, state_tan, node_tan):
        if not self.use_learned_sheaf or self.sheaf_learner is None or self.laplacian_builder is None:
            return None
        maps = self.sheaf_learner(node_tan, self.edge_index)
        laplacian, _ = self.laplacian_builder(maps)
        return torch_sparse.spmm(
            laplacian[0],
            laplacian[1],
            self.graph_size * self.d,
            self.graph_size * self.d,
            -state_tan,
        )

    # 用固定 Hodge Laplacian 构造二阶切向量。
    def _fixed_hodge_tangent(self, node_tan):
        if self.B1 is None or self.hodge_laplacian is None:
            return torch.zeros(
                self.graph_size * self.d,
                self.hidden_channels,
                device=node_tan.device,
                dtype=node_tan.dtype,
            )

        x_edges = self._sparse_mm(self.B1.t(), node_tan)
        v_edges = self._sparse_mm(self.hodge_laplacian, -x_edges)
        return self._sparse_mm(self.B1, v_edges)

    @staticmethod
    def _sparse_mm(matrix, dense):
        if not matrix.is_sparse:
            return matrix @ dense
        matrix = matrix.coalesce()
        values = matrix.values()
        if values.dtype != dense.dtype:
            values = values.to(dtype=dense.dtype)
        return torch_sparse.spmm(
            matrix.indices(), values,
            matrix.size(0), matrix.size(1), dense)

    # 在没有 learnable sheaf 时回退到边聚合扩散。
    def _edge_tangent(self, node_tan):
        src, dst = self.edge_index[0], self.edge_index[1]
        z_src = node_tan[src]
        z_dst = node_tan[dst]
        messages = z_dst - z_src

        if self.edge_attr is not None:
            edge_weights = self.edge_attr.to(messages.device)
        else:
            edge_weights = torch.ones(messages.size(0), 1, device=messages.device)
        edge_weights = torch.clamp(edge_weights, min=0.0, max=10.0)

        return scatter_add(edge_weights * messages, src, dim=0, dim_size=self.graph_size)


class ODEBlock(nn.Module):
    """Module performing ODE integration."""

    def __init__(self, odefunc, t, opt):
        super(ODEBlock, self).__init__()
        self.t = t
        self.opt = opt
        self.odefunc = odefunc
        self.set_tol()
        self.manifold = PoincareBall()

    # 设置 ODE 求解容差。
    def set_tol(self):
        self.atol = self.opt['tol_scale'] * 1e-7
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint']:
            self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

    # 重置 ODE 求解容差。
    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    # 执行 ODE 求解。
    def forward(self, x):
        if self.opt["adjoint"] and self.training:
            z = odeint_adjoint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(
                    step_size=self.opt['step_size'],
                    max_iters=self.opt['max_iters'],
                    manifold=self.manifold,
                ),
                adjoint_method=self.opt['adjoint_method'],
                adjoint_options=dict(
                    step_size=self.opt['adjoint_step_size'],
                    max_iters=self.opt['max_iters'],
                ),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint,
            )
        else:
            z = odeint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(
                    step_size=self.opt['step_size'],
                    max_iters=self.opt['max_iters'],
                    manifold=self.manifold,
                ),
                atol=self.atol,
                rtol=self.rtol,
            )
        self.odefunc.L = None
        return z[1]
