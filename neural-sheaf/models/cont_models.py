import torch
import torch.nn.functional as F
from torch import nn

from lib.hyperbolic_geometry import PoincareBall
from models import laplacian_builders as lb
from models.hyperbolic_ode import HyperbolicLaplacianODEFunc, ODEBlock
from models.sheaf_base import SheafDiffusion
from models.sheaf_models import (
    LocalConcatIncidenceSheafLearner,
    LocalConcatSheafLearner,
)
from utils.continuity_scaling import ACX_SCALE_BANDS
from utils.simplicial import boundary_to_incidence_data


class HyperbolicGraphDiffusion(SheafDiffusion):
    """Diffusion model using hyperbolic sheaf ODE for brain network classification."""

    def __init__(self, edge_index, args):
        super(HyperbolicGraphDiffusion, self).__init__(edge_index, args)
        curvature = args.get('curvature', 1.0)
        learn_curvature = args.get('learn_curvature', False)

        if learn_curvature:
            self.c_param = nn.Parameter(torch.tensor(curvature, dtype=torch.float32))
            self.learn_curvature = True
        else:
            self.register_buffer('c_param', torch.tensor(curvature, dtype=torch.float32))
            self.learn_curvature = False

        self.edge_index = edge_index
        self.pool_type = args.get('pool_type', 'mean')
        self.pool_heads = int(args.get('pool_heads', 4))
        if self.pool_type in (
                'roi_attention', 'mean_roi_attention', 'residual_roi_attention',
                'dual_roi_attention'):
            roi_pool_logits = torch.zeros(self.pool_heads, self.graph_size)
            anchor_strength = float(args.get('roi_attention_init', 0.0))
            if anchor_strength > 0:
                anchor_indices = torch.linspace(
                    0, self.graph_size - 1, steps=self.pool_heads + 2)[1:-1]
                anchor_indices = anchor_indices.round().long()
                roi_pool_logits[
                    torch.arange(self.pool_heads), anchor_indices] = anchor_strength
            self.roi_pool_logits = nn.Parameter(
                roi_pool_logits)
        else:
            self.register_parameter('roi_pool_logits', None)
        if self.pool_type == 'signed_roi_attention':
            projection_generator = torch.Generator(device='cpu')
            projection_generator.manual_seed(1729)
            signed_projection = torch.randn(
                self.pool_heads, self.graph_size,
                generator=projection_generator) * 0.1
            self.roi_signed_logits = nn.Parameter(signed_projection)
        else:
            self.register_parameter('roi_signed_logits', None)
        if self.pool_type == 'residual_roi_attention':
            residual_init = float(args.get('roi_attention_residual_init', 0.1))
            self.roi_pool_residual_logit = nn.Parameter(
                torch.logit(torch.tensor(residual_init, dtype=torch.float32)))
        else:
            self.register_parameter('roi_pool_residual_logit', None)
        self.use_learned_sheaf = args.get('use_learned_sheaf', True)
        self.use_edge_features = (
            args.get('use_edge_features', True)
            and args.get('construction_method', 'pearson') == 'causality'
        )
        self.edge_scale_bands = ACX_SCALE_BANDS if self.use_edge_features else 0
        self.sheaf_edge_channels = self.edge_scale_bands * 2
        self.adaptive_edge_scale_gates = args.get(
            'adaptive_edge_scale_gates', False)
        self.adaptive_edge_scale_strength = args.get(
            'adaptive_edge_scale_strength', 0.1)
        self.use_fixed_hodge = args.get('use_fixed_hodge', True)
        self.hodge_weight = args.get('hodge_weight', 1.0)
        self.diffusion_residual = args.get('diffusion_residual', False)
        residual_init = float(args.get('diffusion_residual_init', 0.5))
        residual_logit = torch.logit(torch.tensor(residual_init, dtype=torch.float32))
        self.diffusion_residual_logit = nn.Parameter(
            residual_logit, requires_grad=self.diffusion_residual)
        self.simplex_order = args.get('simplex_order', 2)
        self.use_learned_hodge = (
            self.simplex_order == 2
            and self.use_learned_sheaf
            and not self.use_fixed_hodge
        )

        self.multiscale_node_fusion = args.get('multiscale_node_fusion', False)
        self.multiscale_node_bands = 0
        encoder_input_dim = self.input_dim
        if self.multiscale_node_fusion:
            feature_blocks = self.input_dim // self.graph_size
            self.multiscale_node_bands = feature_blocks - 1
            fusion_init = float(args.get('multiscale_node_fusion_init', 0.1))
            self.multiscale_node_scale_logits = nn.Parameter(
                torch.zeros(self.multiscale_node_bands))
            self.multiscale_node_fusion_logit = nn.Parameter(
                torch.logit(torch.tensor(fusion_init, dtype=torch.float32)))
            encoder_input_dim = self.graph_size
        else:
            self.register_parameter('multiscale_node_scale_logits', None)
            self.register_parameter('multiscale_node_fusion_logit', None)

        self.lin1 = nn.Linear(encoder_input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.pool_type in ('mean_std', 'mean_max'):
            pooled_dim = self.hidden_dim * 2
        elif self.pool_type == 'roi_attention':
            pooled_dim = self.hidden_dim * self.pool_heads
        elif self.pool_type == 'mean_roi_attention':
            pooled_dim = self.hidden_dim * (self.pool_heads + 1)
        elif self.pool_type == 'signed_roi_attention':
            pooled_dim = self.hidden_dim * self.pool_heads
        else:
            pooled_dim = self.hidden_dim
        classifier_layers = [nn.Linear(pooled_dim, self.hidden_dim)]
        if args.get('classifier_norm', False):
            classifier_layers.append(nn.LayerNorm(self.hidden_dim))
        classifier_layers.extend([
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        ])
        self.classifier = nn.Sequential(*classifier_layers)

        if self.pool_type == 'dual_roi_attention':
            rng_state = torch.random.get_rng_state()
            self.roi_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim * self.pool_heads, self.hidden_dim),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
            torch.random.set_rng_state(rng_state)
            roi_logit_init = float(args.get('roi_logit_fusion_init', 0.05))
            self.roi_logit_fusion_logit = nn.Parameter(
                torch.logit(torch.tensor(roi_logit_init, dtype=torch.float32)))
        else:
            self.roi_classifier = None
            self.register_parameter('roi_logit_fusion_logit', None)

        self.manifold = PoincareBall()
        self.sheaf_learners = nn.ModuleList()
        self.hodge_down_sheaf_learners = nn.ModuleList()
        self.hodge_up_sheaf_learners = nn.ModuleList()
        self.laplacian_builders = nn.ModuleList()
        self.odefuncs = nn.ModuleList()
        self.odeblocks = nn.ModuleList()
        self.laplacian_builder_cache = {}
        self.hodge_laplacian_builder_cache = {}

        ode_opt = self._build_ode_options(args)
        for _ in range(max(1, self.layers)):
            sheaf_learner = LocalConcatSheafLearner(
                self.hidden_dim,
                (self.final_d,),
                self.sheaf_act,
                edge_channels=self.sheaf_edge_channels,
                edge_scale_bands=self.edge_scale_bands,
                adaptive_edge_scale_gates=self.adaptive_edge_scale_gates,
                adaptive_edge_scale_strength=self.adaptive_edge_scale_strength,
            )
            laplacian_builder = lb.DiagLaplacianBuilder(
                self.graph_size,
                edge_index,
                d=self.final_d,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
                add_hp=False,
                add_lp=False,
            )
            if self.use_learned_hodge:
                hodge_down_sheaf_learner = LocalConcatIncidenceSheafLearner(
                    self.hidden_dim, (self.final_d,), self.sheaf_act)
                hodge_up_sheaf_learner = LocalConcatIncidenceSheafLearner(
                    self.hidden_dim, (self.final_d,), self.sheaf_act)
                self.hodge_down_sheaf_learners.append(
                    hodge_down_sheaf_learner)
                self.hodge_up_sheaf_learners.append(
                    hodge_up_sheaf_learner)
            else:
                hodge_down_sheaf_learner = None
                hodge_up_sheaf_learner = None
            odefunc = HyperbolicLaplacianODEFunc(
                self.final_d,
                sheaf_learner,
                laplacian_builder,
                edge_index,
                self.graph_size,
                self.hidden_channels,
                hodge_down_sheaf_learner=hodge_down_sheaf_learner,
                hodge_up_sheaf_learner=hodge_up_sheaf_learner,
                left_weights=self.left_weights,
                right_weights=self.right_weights,
                use_act=args.get('use_ode_act', self.use_act),
                max_nfe=args.get('max_nfe', 1000),
                curvature=curvature,
                learn_curvature=learn_curvature,
                simplex_order=self.simplex_order,
                use_learned_sheaf=self.use_learned_sheaf,
                use_fixed_hodge=self.use_fixed_hodge,
                hodge_weight=self.hodge_weight,
                order2_edge_channels=self.sheaf_edge_channels,
                order2_edge_state=args.get('order2_edge_state', True),
                order2_edge_state_weight=args.get(
                    'order2_edge_state_weight', 0.25),
                order2_unsigned_up_projection=args.get(
                    'order2_unsigned_up_projection', True),
                learn_diffusion_time=args.get('learn_diffusion_time', False),
                diffusion_time=args.get('max_t', 0.1),
                diffusion_time_min=args.get('diffusion_time_min', 0.01),
                diffusion_time_max=args.get('diffusion_time_max', 0.5),
            )
            self.sheaf_learners.append(sheaf_learner)
            self.laplacian_builders.append(laplacian_builder)
            self.odefuncs.append(odefunc)
            self.odeblocks.append(ODEBlock(odefunc, self.time_range, ode_opt))

        self.sheaf_learner = self.sheaf_learners[0]
        self.laplacian_builder = self.laplacian_builders[0]
        self.odefunc = self.odefuncs[0]
        self.odeblock = self.odeblocks[0]

    def _build_ode_options(self, args):
        return {
            'int_method': args.get('int_method', 'hyperbolic_rk4'),
            'step_size': args.get('step_size', 0.5),
            'max_iters': args.get('max_nfe', 1000),
            'adjoint': args.get('adjoint', False),
            'adjoint_method': args.get('adjoint_method', 'rk4'),
            'adjoint_step_size': args.get('adjoint_step_size', 1.0),
            'tol_scale': args.get('tol_scale', 1.0),
            'tol_scale_adjoint': args.get('tol_scale_adjoint', 1.0),
        }

    def get_curvature(self):
        return F.softplus(self.c_param)

    def get_diffusion_times(self):
        """Expose per-layer effective times for training diagnostics."""
        return [odefunc.get_diffusion_times() for odefunc in self.odefuncs]

    def get_diffusion_residual_weight(self):
        """Return the learned contribution of the post-diffusion state."""
        if not self.diffusion_residual:
            return None
        return torch.sigmoid(self.diffusion_residual_logit)

    def get_edge_scale_gates(self):
        """Return the most recent mean ACX scale multipliers per layer."""
        return [
            learner.last_edge_scale_gates
            for learner in self.sheaf_learners
            if learner.last_edge_scale_gates is not None
        ]

    def set_adaptive_edge_scale_strength(self, strength):
        """Set the bounded per-edge scale residual used by every diffusion layer."""
        for learner in self.sheaf_learners:
            learner.adaptive_edge_scale_strength = strength

    def get_adaptive_edge_scale_strength(self):
        if not self.adaptive_edge_scale_gates or not self.sheaf_learners:
            return None
        return self.sheaf_learners[0].adaptive_edge_scale_strength

    def get_multiscale_node_fusion(self):
        if not self.multiscale_node_fusion:
            return None
        return {
            'alpha': torch.sigmoid(self.multiscale_node_fusion_logit),
            'weights': torch.softmax(self.multiscale_node_scale_logits, dim=0),
        }

    def get_roi_pool_weights(self):
        if self.roi_pool_logits is None:
            return None
        return torch.softmax(self.roi_pool_logits, dim=-1)

    def get_roi_pool_residual(self):
        if self.roi_pool_residual_logit is None:
            return None
        return torch.sigmoid(self.roi_pool_residual_logit)

    def get_roi_logit_fusion(self):
        if self.roi_logit_fusion_logit is None:
            return None
        return torch.sigmoid(self.roi_logit_fusion_logit)

    def get_signed_roi_weights(self):
        if self.roi_signed_logits is None:
            return None
        weights = torch.tanh(self.roi_signed_logits)
        return weights / weights.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @property
    def c(self):
        return self.get_curvature()

    def forward(self, data):
        for odefunc in self.odefuncs:
            odefunc.nfe = 0

        batch_size = int(data.batch.max().item() + 1) if hasattr(data, 'batch') else 1
        x = data.x
        if self.multiscale_node_fusion:
            feature_blocks = x.view(
                x.size(0), self.multiscale_node_bands + 1, self.graph_size)
            aggregate = feature_blocks[:, 0]
            scale_weights = torch.softmax(
                self.multiscale_node_scale_logits, dim=0)
            scale_specific = torch.sum(
                feature_blocks[:, 1:] * scale_weights.view(1, -1, 1), dim=1)
            fusion_alpha = torch.sigmoid(self.multiscale_node_fusion_logit)
            x = (1.0 - fusion_alpha) * aggregate + fusion_alpha * scale_specific
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        x = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c),
            c=self.c,
        )
        x = x.view(batch_size, self.graph_size, -1)
        x_initial = x

        if self.t > 0:
            x = self._run_ode_layers(data, x, batch_size)

        x = self.manifold.logmap0(x.view(-1, x.size(-1)), self.c)
        x = x.view(batch_size, self.graph_size, -1)
        if self.diffusion_residual:
            x_initial = self.manifold.logmap0(
                x_initial.view(-1, x_initial.size(-1)), self.c)
            x_initial = x_initial.view(batch_size, self.graph_size, -1)
            diffusion_weight = torch.sigmoid(self.diffusion_residual_logit)
            x = (1.0 - diffusion_weight) * x_initial + diffusion_weight * x
        if self.pool_type == 'dual_roi_attention':
            global_logits = self.classifier(x.mean(dim=1))
            weights = torch.softmax(self.roi_pool_logits, dim=-1)
            roi_features = torch.einsum('hn,bnd->bhd', weights, x)
            roi_logits = self.roi_classifier(
                roi_features.flatten(start_dim=1))
            fusion_weight = torch.sigmoid(self.roi_logit_fusion_logit)
            logits = global_logits + fusion_weight * roi_logits
            return F.log_softmax(logits, dim=1)
        x = self._pool_graph(x)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)

    def _run_ode_layers(self, data, x, batch_size):
        if self.odefunc.simplex_order != 2:
            return self._run_batched_order1_ode(data, x, batch_size)

        x_out = []
        for sample_idx in range(batch_size):
            edge_index, edge_attr, sheaf_edge_attr = self._get_sample_edges(
                data, sample_idx, x.device)
            B1, B2 = self._get_boundary_matrices(data, sample_idx, x.device)
            incidence_context = self._get_incidence_context(
                data, sample_idx, x.device, B1, B2)
            x_single = x[sample_idx].view(self.graph_size * self.final_d, -1)

            for layer_idx, (odefunc, odeblock) in enumerate(zip(self.odefuncs, self.odeblocks)):
                cache_key = self._laplacian_cache_key(data, sample_idx, layer_idx, edge_index)
                laplacian_builder = self._get_laplacian_builder(cache_key, self.graph_size, edge_index)
                hodge_down_builder = None
                hodge_up_builder = None
                if self.use_learned_hodge:
                    edge_node_index, edge_node_sign, face_edge_index, face_edge_sign = (
                        incidence_context)
                    hodge_down_builder = self._get_hodge_laplacian_builder(
                        self._hodge_laplacian_cache_key(
                            cache_key, 'hodge_down',
                            edge_node_index, edge_node_sign),
                        B1.size(1),
                        edge_node_index,
                        edge_node_sign,
                        edge_dim=0,
                    )
                    hodge_up_builder = self._get_hodge_laplacian_builder(
                        self._hodge_laplacian_cache_key(
                            cache_key, 'hodge_up',
                            face_edge_index, face_edge_sign),
                        B1.size(1),
                        face_edge_index,
                        face_edge_sign,
                        edge_dim=1,
                    )
                self.laplacian_builders[layer_idx] = laplacian_builder
                self.sheaf_learners[layer_idx].edge_index = edge_index
                odefunc.set_graph_context(
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    sheaf_edge_attr=sheaf_edge_attr,
                    B1=B1,
                    B2=B2,
                    edge_node_index=incidence_context[0],
                    edge_node_sign=incidence_context[1],
                    face_edge_index=incidence_context[2],
                    face_edge_sign=incidence_context[3],
                    laplacian_builder=laplacian_builder,
                    hodge_down_laplacian_builder=hodge_down_builder,
                    hodge_up_laplacian_builder=hodge_up_builder,
                )
                x_single = odeblock(x_single)

            x_out.append(x_single.view(self.graph_size, -1))
        return torch.stack(x_out, dim=0)

    def _get_laplacian_builder(self, cache_key, graph_size, edge_index):
        builder = self.laplacian_builder_cache.get(cache_key)
        if builder is None:
            builder = lb.DiagLaplacianBuilder(
                graph_size,
                edge_index,
                d=self.final_d,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
                add_hp=False,
                add_lp=False,
            )
            self.laplacian_builder_cache[cache_key] = builder
        return builder

    def _get_hodge_laplacian_builder(
            self, cache_key, edge_size, incidence_index, incidence_sign,
            edge_dim):
        builder = self.hodge_laplacian_builder_cache.get(cache_key)
        if builder is None:
            builder = lb.DiagHodgeLaplacianBuilder(
                edge_size,
                incidence_index,
                incidence_sign,
                d=self.final_d,
                edge_dim=edge_dim,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
            )
            self.hodge_laplacian_builder_cache[cache_key] = builder
        return builder

    @staticmethod
    def _hodge_laplacian_cache_key(
            base_key, kind, incidence_index, incidence_sign):
        if base_key and base_key[0] == 'sample':
            return base_key + (kind,)
        index_bytes = incidence_index.detach().cpu().contiguous().numpy().tobytes()
        sign_bytes = incidence_sign.detach().cpu().contiguous().numpy().tobytes()
        return base_key + (kind, index_bytes, sign_bytes)

    def _laplacian_cache_key(self, data, sample_idx, layer_idx, edge_index):
        if hasattr(data, 'graph_idx'):
            graph_idx = int(data.graph_idx[sample_idx].item())
            return ('sample', graph_idx, layer_idx, str(edge_index.device))
        edge_bytes = edge_index.detach().cpu().contiguous().numpy().tobytes()
        return ('edge_index', edge_bytes, layer_idx, str(edge_index.device))

    def _run_batched_order1_ode(self, data, x, batch_size):
        device = x.device
        total_nodes = batch_size * self.graph_size
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        sheaf_edge_attr = None
        if self.use_edge_features and hasattr(data, 'sheaf_edge_attr') and data.sheaf_edge_attr is not None:
            sheaf_edge_attr = data.sheaf_edge_attr.to(device)
        x_batch = x.reshape(total_nodes * self.final_d, -1)

        laplacian_builder = None
        if self.use_learned_sheaf:
            laplacian_builder = lb.DiagLaplacianBuilder(
                total_nodes,
                edge_index,
                d=self.final_d,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
                add_hp=False,
                add_lp=False,
            )

        for layer_idx, (odefunc, odeblock) in enumerate(zip(self.odefuncs, self.odeblocks)):
            if laplacian_builder is not None:
                self.laplacian_builders[layer_idx] = laplacian_builder
            self.sheaf_learners[layer_idx].edge_index = edge_index
            odefunc.set_graph_context(
                edge_index=edge_index,
                edge_attr=edge_attr,
                sheaf_edge_attr=sheaf_edge_attr,
                B1=None,
                B2=None,
                edge_node_index=None,
                edge_node_sign=None,
                face_edge_index=None,
                face_edge_sign=None,
                laplacian_builder=laplacian_builder,
                hodge_down_laplacian_builder=None,
                hodge_up_laplacian_builder=None,
                graph_size=total_nodes,
            )
            x_batch = odeblock(x_batch)

        return x_batch.view(batch_size, self.graph_size, -1)

    def _get_sample_edges(self, data, sample_idx, device):
        if not hasattr(data, 'batch') or not hasattr(data, 'ptr'):
            edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            sheaf_edge_attr = None
            if self.use_edge_features and hasattr(data, 'sheaf_edge_attr') and data.sheaf_edge_attr is not None:
                sheaf_edge_attr = data.sheaf_edge_attr.to(device)
            return data.edge_index.to(device), edge_attr, sheaf_edge_attr

        start = int(data.ptr[sample_idx].item())
        end = int(data.ptr[sample_idx + 1].item())
        edge_index = data.edge_index
        mask = (edge_index[0] >= start) & (edge_index[0] < end)
        sample_edge_index = (edge_index[:, mask] - start).to(device)
        sample_edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            sample_edge_attr = data.edge_attr[mask].to(device)
        sample_sheaf_edge_attr = None
        if self.use_edge_features and hasattr(data, 'sheaf_edge_attr') and data.sheaf_edge_attr is not None:
            sample_sheaf_edge_attr = data.sheaf_edge_attr[mask].to(device)
        return sample_edge_index, sample_edge_attr, sample_sheaf_edge_attr

    def _get_boundary_matrices(self, data, sample_idx, device):
        if self.odefunc.simplex_order != 2:
            return None, None

        if hasattr(data, 'sigma1_list') and hasattr(data, 'sigma2_list'):
            B1 = data.sigma1_list[sample_idx]
            B2 = data.sigma2_list[sample_idx]
        elif hasattr(data, 'sigma1') and hasattr(data, 'sigma2'):
            B1 = data.sigma1
            B2 = data.sigma2
        B1 = B1.to(device)
        B2 = B2.to(device)
        if B1.size(0) != self.graph_size:
            B1 = B1[:self.graph_size, :]
        if B2.size(0) != B1.size(1):
            B2 = B2[:B1.size(1), :]
        return B1, B2

    def _get_incidence_context(self, data, sample_idx, device, B1, B2):
        if self.odefunc.simplex_order != 2:
            return None, None, None, None

        list_attributes = (
            'edge_node_index_list',
            'edge_node_sign_list',
            'face_edge_index_list',
            'face_edge_sign_list',
        )
        single_attributes = (
            'edge_node_index',
            'edge_node_sign',
            'face_edge_index',
            'face_edge_sign',
        )
        if all(hasattr(data, name) for name in list_attributes):
            values = tuple(
                getattr(data, name)[sample_idx] for name in list_attributes)
        elif all(hasattr(data, name) for name in single_attributes):
            values = tuple(getattr(data, name) for name in single_attributes)
        else:
            edge_node_index, edge_node_sign = boundary_to_incidence_data(B1)
            face_edge_index, face_edge_sign = boundary_to_incidence_data(B2)
            values = (
                edge_node_index,
                edge_node_sign,
                face_edge_index,
                face_edge_sign,
            )

        return tuple(value.to(device) for value in values)

    def _pool_graph(self, x):
        if self.pool_type == 'mean':
            return x.mean(dim=1)
        if self.pool_type == 'max':
            return x.max(dim=1)[0]
        if self.pool_type == 'sum':
            return x.sum(dim=1)
        if self.pool_type == 'attention':
            attn_weights = torch.softmax(x.mean(dim=-1), dim=-1)
            return (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        if self.pool_type == 'mean_std':
            return torch.cat([
                x.mean(dim=1),
                x.std(dim=1, unbiased=False),
            ], dim=-1)
        if self.pool_type == 'mean_max':
            return torch.cat([
                x.mean(dim=1),
                x.max(dim=1).values,
            ], dim=-1)
        if self.pool_type == 'signed_roi_attention':
            weights = self.get_signed_roi_weights()
            pooled = torch.einsum('hn,bnd->bhd', weights, x)
            return pooled.flatten(start_dim=1)
        if self.pool_type in (
                'roi_attention', 'mean_roi_attention', 'residual_roi_attention'):
            weights = torch.softmax(self.roi_pool_logits, dim=-1)
            pooled = torch.einsum('hn,bnd->bhd', weights, x)
            if self.pool_type == 'residual_roi_attention':
                roi_summary = pooled.mean(dim=1)
                global_mean = x.mean(dim=1)
                residual_weight = torch.sigmoid(
                    self.roi_pool_residual_logit)
                return global_mean + residual_weight * (
                    roi_summary - global_mean)
            pooled = pooled.flatten(start_dim=1)
            if self.pool_type == 'mean_roi_attention':
                return torch.cat([x.mean(dim=1), pooled], dim=-1)
            return pooled
        return x.mean(dim=1)
