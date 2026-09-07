import torch
import torch.nn.functional as F
from torch import nn

from lib.hyperbolic_geometry import PoincareBall
from models import laplacian_builders as lb
from models.hyperbolic_ode import HyperbolicLaplacianODEFunc, ODEBlock
from models.sheaf_base import SheafDiffusion
from models.sheaf_models import LocalConcatSheafLearner


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
        self.use_learned_sheaf = args.get('use_learned_sheaf', True)
        self.use_fixed_hodge = args.get('use_fixed_hodge', True)
        self.hodge_weight = args.get('hodge_weight', 1.0)

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.manifold = PoincareBall()
        self.sheaf_learners = nn.ModuleList()
        self.laplacian_builders = nn.ModuleList()
        self.odefuncs = nn.ModuleList()
        self.odeblocks = nn.ModuleList()
        self.laplacian_builder_cache = {}

        ode_opt = self._build_ode_options(args)
        for _ in range(max(1, self.layers)):
            sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, (self.final_d,), self.sheaf_act)
            laplacian_builder = lb.DiagLaplacianBuilder(
                self.graph_size,
                edge_index,
                d=self.final_d,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
                add_hp=False,
                add_lp=False,
            )
            odefunc = HyperbolicLaplacianODEFunc(
                self.final_d,
                sheaf_learner,
                laplacian_builder,
                edge_index,
                self.graph_size,
                self.hidden_channels,
                left_weights=self.left_weights,
                right_weights=self.right_weights,
                use_act=self.use_act,
                max_nfe=args.get('max_nfe', 1000),
                curvature=curvature,
                learn_curvature=learn_curvature,
                simplex_order=args.get('simplex_order', 2),
                use_learned_sheaf=self.use_learned_sheaf,
                use_fixed_hodge=self.use_fixed_hodge,
                hodge_weight=self.hodge_weight,
            )
            self.sheaf_learners.append(sheaf_learner)
            self.laplacian_builders.append(laplacian_builder)
            self.odefuncs.append(odefunc)
            self.odeblocks.append(ODEBlock(odefunc, self.time_range, ode_opt))

        self.sheaf_learner = self.sheaf_learners[0]
        self.laplacian_builder = self.laplacian_builders[0]
        self.odefunc = self.odefuncs[0]
        self.odeblock = self.odeblocks[0]

    # 构造 ODE 求解参数。
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

    # 获取当前双曲曲率。
    def get_curvature(self):
        return F.softplus(self.c_param)

    # 返回双曲曲率。
    @property
    def c(self):
        return self.get_curvature()

    # 执行图分类前向计算。
    def forward(self, data):
        for odefunc in self.odefuncs:
            odefunc.nfe = 0

        batch_size = int(data.num_graphs) if hasattr(data, 'num_graphs') else 1
        x = data.x
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

        if self.t > 0:
            x = self._run_ode_layers(data, x, batch_size)

        x = self.manifold.logmap0(x.view(-1, x.size(-1)), self.c)
        x = x.view(batch_size, self.graph_size, -1)
        x = self._pool_graph(x)
        return F.log_softmax(self.classifier(x), dim=1)

    # 对 batch 内每个样本分别执行多层 HPDE。
    def _run_ode_layers(self, data, x, batch_size):
        # A PyG batch is a disjoint union of graphs. Its learned sheaf
        # Laplacian is block diagonal, so order-1 graphs can be integrated in
        # one ODE solve without changing the per-graph computation.
        if self.odefunc.simplex_order != 2:
            return self._run_batched_order1_ode(data, x, batch_size)

        x_out = []
        for sample_idx in range(batch_size):
            edge_index, edge_attr = self._get_sample_edges(data, sample_idx, x.device)
            B1, hodge_laplacian = self._get_hodge_operators(
                data, sample_idx, x.device, x.dtype)
            x_single = x[sample_idx].view(self.graph_size * self.final_d, -1)

            # All diffusion layers for a sample use the same topology. Reuse
            # one builder rather than caching duplicate copies per layer.
            cache_key = self._laplacian_cache_key(data, sample_idx, edge_index)
            laplacian_builder = self._get_laplacian_builder(
                cache_key, self.graph_size, edge_index)

            for layer_idx, (odefunc, odeblock) in enumerate(zip(self.odefuncs, self.odeblocks)):
                self.laplacian_builders[layer_idx] = laplacian_builder
                self.sheaf_learners[layer_idx].edge_index = edge_index
                odefunc.set_graph_context(
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    B1=B1,
                    hodge_laplacian=hodge_laplacian,
                    laplacian_builder=laplacian_builder,
                )
                x_single = odeblock(x_single)

            x_out.append(x_single.view(self.graph_size, -1))
        return torch.stack(x_out, dim=0)

    # 复用同一样本、同一层的 LaplacianBuilder 索引。
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

    # 优先使用数据集索引作为缓存键，缺失时退回到 edge_index 内容。
    def _laplacian_cache_key(self, data, sample_idx, edge_index):
        if hasattr(data, 'graph_idx'):
            graph_idx = int(data.graph_idx[sample_idx].item())
            return ('sample', graph_idx, str(edge_index.device))
        edge_bytes = edge_index.detach().cpu().contiguous().numpy().tobytes()
        return ('edge_index', edge_bytes, str(edge_index.device))

    # 一阶图使用 PyG batch 合并后的 disjoint graph 一次性执行 HPDE。
    def _run_batched_order1_ode(self, data, x, batch_size):
        device = x.device
        total_nodes = batch_size * self.graph_size
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        x_batch = x.reshape(total_nodes * self.final_d, -1)

        # The topology is identical for every diffusion layer in this forward
        # pass. Build its constant sparse indices once and share them across
        # layers; only the learned restriction-map values change per layer/NFE.
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
                B1=None,
                hodge_laplacian=None,
                laplacian_builder=laplacian_builder,
                graph_size=total_nodes,
            )
            x_batch = odeblock(x_batch)

        return x_batch.view(batch_size, self.graph_size, -1)

    # 取出 batch 中某个样本的边结构。
    def _get_sample_edges(self, data, sample_idx, device):
        if not hasattr(data, 'batch') or not hasattr(data, 'ptr'):
            edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            return data.edge_index.to(device), edge_attr

        start = sample_idx * self.graph_size
        end = start + self.graph_size
        edge_index = data.edge_index
        mask = (edge_index[0] >= start) & (edge_index[0] < end)
        sample_edge_index = (edge_index[:, mask] - start).to(device)
        sample_edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            sample_edge_attr = data.edge_attr[mask].to(device)
        return sample_edge_index, sample_edge_attr

    # 取出 batch 中某个样本的边界矩阵。
    def _get_hodge_operators(self, data, sample_idx, device, dtype):
        if self.odefunc.simplex_order != 2:
            return None, None

        if hasattr(data, 'sigma1_list') and hasattr(data, 'hodge_laplacian_list'):
            B1 = data.sigma1_list[sample_idx]
            hodge_laplacian = data.hodge_laplacian_list[sample_idx]
        elif hasattr(data, 'sigma1') and hasattr(data, 'hodge_laplacian'):
            B1 = data.sigma1
            hodge_laplacian = data.hodge_laplacian
        else:
            raise ValueError(
                "Order-2 data must contain sigma1 and a precomputed Hodge Laplacian")

        B1 = B1.to(device=device, dtype=dtype)
        hodge_laplacian = hodge_laplacian.to(device=device, dtype=dtype)
        if B1.size(0) != self.graph_size:
            raise ValueError(f"sigma1 has {B1.size(0)} nodes; expected {self.graph_size}")
        if hodge_laplacian.size(0) != B1.size(1) or hodge_laplacian.size(1) != B1.size(1):
            raise ValueError("Hodge Laplacian size must equal the number of edges in sigma1")
        return B1, hodge_laplacian

    # 聚合节点表示为图表示。
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
        return x.mean(dim=1)

    # 更新模型中的默认边索引。
    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.edge_index = edge_index
        for odefunc in self.odefuncs:
            odefunc.edge_index = edge_index
        for sheaf_learner in self.sheaf_learners:
            sheaf_learner.edge_index = edge_index
