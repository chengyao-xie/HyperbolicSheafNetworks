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
                 hodge_down_sheaf_learner=None,
                 hodge_up_sheaf_learner=None,
                 left_weights=False,
                 right_weights=False,
                 use_act=False,
                 weight_learner=None,
                 edge_attr=None,
                 sheaf_edge_attr=None,
                 max_nfe=1000,
                 curvature=1.0,
                 learn_curvature=False,
                 simplex_order=2,
                 use_learned_sheaf=True,
                 use_fixed_hodge=True,
                 hodge_weight=1.0,
                 order2_edge_channels=0,
                 order2_edge_state=True,
                 order2_edge_state_weight=0.25,
                 order2_unsigned_up_projection=True,
                 learn_diffusion_time=False,
                 diffusion_time=0.1,
                 diffusion_time_min=0.01,
                 diffusion_time_max=0.5):
        super(HyperbolicLaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner
        self.laplacian_builder = laplacian_builder
        self.hodge_down_sheaf_learner = hodge_down_sheaf_learner
        self.hodge_up_sheaf_learner = hodge_up_sheaf_learner
        self.hodge_down_laplacian_builder = None
        self.hodge_up_laplacian_builder = None
        self.edge_index = edge_index
        self.edge_attr = edge_attr if edge_attr is not None else torch.ones(edge_index.size(1), 1)
        self.sheaf_edge_attr = sheaf_edge_attr
        self.B1 = None
        self.B2 = None
        self.edge_node_index = None
        self.edge_node_sign = None
        self.face_edge_index = None
        self.face_edge_sign = None
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        self.simplex_order = simplex_order
        self.use_learned_sheaf = use_learned_sheaf
        self.use_fixed_hodge = use_fixed_hodge
        self.hodge_weight = hodge_weight
        self.order2_edge_state = bool(order2_edge_state)
        self.order2_edge_state_weight = float(order2_edge_state_weight)
        self.order2_unsigned_up_projection = bool(
            order2_unsigned_up_projection)
        if (self.simplex_order == 2 and self.order2_edge_state
                and order2_edge_channels > 0):
            self.order2_edge_encoder = nn.Linear(
                int(order2_edge_channels) * 2,
                self.d * self.hidden_channels,
                bias=False,
            )
            nn.init.xavier_uniform_(self.order2_edge_encoder.weight)
        else:
            self.order2_edge_encoder = None
        self.learn_diffusion_time = learn_diffusion_time
        self.base_diffusion_time = max(float(diffusion_time), 1e-8)
        self.diffusion_time_min = float(diffusion_time_min)
        self.diffusion_time_max = float(diffusion_time_max)
        time_names = ['main']
        if self.simplex_order == 2 and not self.use_fixed_hodge:
            time_names.extend(['down', 'up'])
        self.diffusion_time_logits = nn.ParameterDict()
        initial_logit = self._initial_diffusion_time_logit(
            self.base_diffusion_time)
        for name in time_names:
            parameter = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
            parameter.requires_grad_(self.learn_diffusion_time)
            self.diffusion_time_logits[name] = parameter

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

    def set_graph_context(self, edge_index, edge_attr=None, sheaf_edge_attr=None,
                          B1=None, B2=None,
                          edge_node_index=None, edge_node_sign=None,
                          face_edge_index=None, face_edge_sign=None,
                          laplacian_builder=None,
                          hodge_down_laplacian_builder=None,
                          hodge_up_laplacian_builder=None,
                          graph_size=None):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.sheaf_edge_attr = sheaf_edge_attr
        self.B1 = B1
        self.B2 = B2
        self.edge_node_index = edge_node_index
        self.edge_node_sign = edge_node_sign
        self.face_edge_index = face_edge_index
        self.face_edge_sign = face_edge_sign
        if graph_size is not None:
            self.graph_size = graph_size
        if laplacian_builder is not None:
            self.laplacian_builder = laplacian_builder
        self.hodge_down_laplacian_builder = hodge_down_laplacian_builder
        self.hodge_up_laplacian_builder = hodge_up_laplacian_builder

    def get_curvature(self):
        return F.softplus(self.c_param)

    def _initial_diffusion_time_logit(self, value):
        span = self.diffusion_time_max - self.diffusion_time_min
        probability = (float(value) - self.diffusion_time_min) / span
        probability = min(max(probability, 1e-4), 1.0 - 1e-4)
        return float(torch.logit(torch.tensor(probability)).item())

    def get_diffusion_times(self):
        """Return bounded effective diffusion times for each Hodge channel."""
        if not self.learn_diffusion_time:
            return {
                name: logit.new_tensor(self.base_diffusion_time)
                for name, logit in self.diffusion_time_logits.items()
            }
        span = self.diffusion_time_max - self.diffusion_time_min
        return {
            name: self.diffusion_time_min + span * torch.sigmoid(logit)
            for name, logit in self.diffusion_time_logits.items()
        }

    def _diffusion_time_ratio(self, name):
        return self.get_diffusion_times()[name] / self.base_diffusion_time

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
        main_time_ratio = self._diffusion_time_ratio('main')
        tangent = main_time_ratio * tangent

        if self.simplex_order == 2:
            if self.use_fixed_hodge:
                hodge_tangent = self._fixed_hodge_tangent(node_tan)
                if hodge_tangent is not None:
                    hodge_tangent = main_time_ratio * hodge_tangent
            else:
                hodge_terms = self._learned_sheaf_tangent_order2(
                    state_tan, node_tan)
                hodge_tangent = None
                if hodge_terms is not None:
                    down_tangent, up_tangent = hodge_terms
                    hodge_tangent = (
                        self._diffusion_time_ratio('down') * down_tangent
                        + self._diffusion_time_ratio('up') * up_tangent
                    )
            if hodge_tangent is not None:
                tangent = (
                    tangent
                    + self.hodge_weight * hodge_tangent.view_as(state_tan))

        tangent = torch.where(torch.isfinite(tangent), tangent, torch.zeros_like(tangent))
        if self.use_act:
            tangent = self.sigma(tangent)

        next_state = self.manifold.expmap(tangent, state, c=curvature)
        next_state = self.manifold.proj(next_state, c=curvature)
        return next_state

    def _check_nfe(self):
        if self.nfe > self.max_nfe:
            raise MaxNFEException
        self.nfe += 1

    def _as_state(self, x):
        if x.dim() == 2 and x.size(0) == self.graph_size * self.d:
            return x
        if x.dim() == 2 and x.size(0) == self.graph_size:
            return x.view(self.graph_size * self.d, -1)
        return x.view(self.graph_size * self.d, -1)

    def _apply_channel_weights(self, state_tan):
        x = state_tan
        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, self.graph_size * self.d).t()
        if self.right_weights:
            x = self.lin_right_weights(x)
        return x

    def _learned_sheaf_tangent(self, state_tan, node_tan):
        if not self.use_learned_sheaf or self.sheaf_learner is None or self.laplacian_builder is None:
            return None
        maps = self.sheaf_learner(
            node_tan, self.edge_index, self.sheaf_edge_attr)
        laplacian, _ = self.laplacian_builder(maps)
        return torch_sparse.spmm(
            laplacian[0],
            laplacian[1],
            self.graph_size * self.d,
            self.graph_size * self.d,
            -state_tan,
        )

    def _learned_sheaf_tangent_order2(self, state_tan, node_tan):
        """Apply learned lower and upper Hodge sheaf terms in edge space."""
        del state_tan  # The edge cochain is induced from the node tangent below.
        required = (
            self.use_learned_sheaf,
            self.B1 is not None,
            self.B2 is not None,
            self.edge_node_index is not None,
            self.face_edge_index is not None,
            self.hodge_down_sheaf_learner is not None,
            self.hodge_up_sheaf_learner is not None,
            self.hodge_down_laplacian_builder is not None,
            self.hodge_up_laplacian_builder is not None,
        )
        if not all(required):
            return None

        edge_features = self._sparse_mm(self.B1.t(), node_tan)
        edge_features = self._augment_order2_edge_state(edge_features)
        face_features = self._unsigned_boundary_mean(self.B2, edge_features)

        maps_down = self.hodge_down_sheaf_learner(
            edge_features, node_tan, self.edge_node_index)
        laplacian_down = self.hodge_down_laplacian_builder(maps_down)

        maps_up = self.hodge_up_sheaf_learner(
            face_features, edge_features, self.face_edge_index)
        laplacian_up = self.hodge_up_laplacian_builder(maps_up)

        edge_state = edge_features.view(
            edge_features.size(0) * self.d, self.hidden_channels)
        down_tangent = self._spmm(laplacian_down, -edge_state)
        up_tangent = self._spmm(laplacian_up, -edge_state)
        down_tangent = down_tangent.view_as(edge_features)
        up_tangent = up_tangent.view_as(edge_features)
        if self.order2_unsigned_up_projection:
            # A signed B1 projection suppresses the upper signal through the
            # boundary-of-a-boundary identity B1 B2 = 0.  For graph-level
            # classification we use an orientation-invariant incidence mean,
            # retaining face-derived energy at its incident nodes.
            up_node_tangent = self._unsigned_upper_to_lower_mean(
                self.B1, up_tangent)
        else:
            up_node_tangent = self._sparse_mm(self.B1, up_tangent)
        return self._sparse_mm(self.B1, down_tangent), up_node_tangent

    def _augment_order2_edge_state(self, edge_features):
        """Add directed ACX restriction attributes to the edge cochain."""
        if (self.order2_edge_encoder is None
                or self.sheaf_edge_attr is None
                or self.order2_edge_state_weight == 0):
            return edge_features
        attributes = self.sheaf_edge_attr.to(
            device=edge_features.device, dtype=edge_features.dtype)
        edge_count = edge_features.size(0)
        if attributes.dim() != 2 or attributes.size(0) != edge_count * 2:
            return edge_features
        directed_pair = attributes.reshape(edge_count, -1)
        encoded = torch.tanh(self.order2_edge_encoder(directed_pair))
        return edge_features + self.order2_edge_state_weight * encoded

    def _fixed_hodge_tangent(self, node_tan):
        if self.B1 is None or self.B2 is None:
            return torch.zeros(
                self.graph_size * self.d,
                self.hidden_channels,
                device=node_tan.device,
                dtype=node_tan.dtype,
            )

        laplacian = self.B2 @ self.B2.t() + self.B1.t() @ self.B1
        laplacian = laplacian.to_sparse()
        x_edges = self.B1.t() @ node_tan
        v_edges = torch_sparse.spmm(
            laplacian.indices(),
            laplacian.values(),
            x_edges.size(0),
            x_edges.size(0),
            -x_edges,
        )
        return self.B1 @ v_edges

    @staticmethod
    def _spmm(laplacian, dense):
        indices, values = laplacian
        return torch_sparse.spmm(
            indices,
            values.to(dtype=dense.dtype),
            dense.size(0),
            dense.size(0),
            dense,
        )

    @staticmethod
    def _sparse_mm(matrix, dense):
        if not matrix.is_sparse:
            return matrix.to(dtype=dense.dtype) @ dense
        matrix = matrix.coalesce()
        return torch_sparse.spmm(
            matrix.indices(),
            matrix.values().to(dtype=dense.dtype),
            matrix.size(0),
            matrix.size(1),
            dense,
        )

    @classmethod
    def _unsigned_boundary_mean(cls, boundary, lower_features):
        if boundary.size(1) == 0:
            return lower_features.new_zeros((0, lower_features.size(1)))

        if boundary.is_sparse:
            boundary = boundary.coalesce()
            unsigned = torch.sparse_coo_tensor(
                boundary.indices(),
                boundary.values().abs(),
                boundary.size(),
                device=boundary.device,
                dtype=boundary.dtype,
            ).coalesce()
            summed = cls._sparse_mm(unsigned.t(), lower_features)
            degree = cls._sparse_mm(
                unsigned.t(),
                lower_features.new_ones((unsigned.size(0), 1)),
            ).t()
        else:
            unsigned = boundary.abs().to(dtype=lower_features.dtype)
            summed = unsigned.t() @ lower_features
            degree = unsigned.sum(dim=0, keepdim=True)
        return summed / degree.t().clamp_min(1)

    @classmethod
    def _unsigned_upper_to_lower_mean(cls, boundary, upper_features):
        """Project upper-simplex features down without orientation cancellation."""
        if boundary.size(1) == 0:
            return upper_features.new_zeros(
                (boundary.size(0), upper_features.size(1)))
        if boundary.is_sparse:
            boundary = boundary.coalesce()
            unsigned = torch.sparse_coo_tensor(
                boundary.indices(),
                boundary.values().abs(),
                boundary.size(),
                device=boundary.device,
                dtype=boundary.dtype,
            ).coalesce()
            summed = cls._sparse_mm(unsigned, upper_features)
            degree = cls._sparse_mm(
                unsigned,
                upper_features.new_ones((unsigned.size(1), 1)),
            )
        else:
            unsigned = boundary.abs().to(dtype=upper_features.dtype)
            summed = unsigned @ upper_features
            degree = unsigned.sum(dim=1, keepdim=True)
        return summed / degree.clamp_min(1)

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
