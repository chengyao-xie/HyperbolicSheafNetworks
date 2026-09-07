# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import lib.laplace as lap

from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import degree


class LaplacianBuilder(nn.Module):

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, add_hp=False, add_lp=False,
                 augmented=True):
        super(LaplacianBuilder, self).__init__()
        self.d = d
        self.final_d = d
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1
        self.size = size
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.device = edge_index.device
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.augmented = augmented

        self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(edge_index)
        if self.add_lp or self.add_hp:
            self.fixed_diag_indices, self.fixed_tril_indices = lap.compute_fixed_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.final_d)
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

    def get_fixed_maps(self, size, dtype):
        fixed_diag, fixed_non_diag = [], []
        if self.add_lp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(torch.ones(size=(size, 1), device=self.device, dtype=dtype))
        if self.add_hp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(-torch.ones(size=(size, 1), device=self.device, dtype=dtype))

        fixed_diag = torch.cat(fixed_diag, dim=1)
        fixed_non_diag = torch.cat(fixed_non_diag, dim=1)

        return fixed_diag, fixed_non_diag

    def scalar_normalise(self, diag, tril, row, col):
        d = diag.size(-1)

        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float('inf'), 0)
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps

    def append_fixed_maps(self, size, diag_indices, diag_maps, tril_indices, tril_maps):
        if not self.add_lp and not self.add_hp:
            return (diag_indices, diag_maps), (tril_indices, tril_maps)

        fixed_diag, fixed_non_diag = self.get_fixed_maps(size, tril_maps.dtype)
        tril_row, tril_col = self.vertex_tril_idx

        if self.normalised:
            fixed_diag, fixed_non_diag = self.scalar_normalise(fixed_diag, fixed_non_diag, tril_row, tril_col)
        fixed_diag, fixed_non_diag = fixed_diag.view(-1), fixed_non_diag.view(-1)
        tril_indices, tril_maps = lap.mergesp(self.fixed_tril_indices, fixed_non_diag, tril_indices, tril_maps)
        diag_indices, diag_maps = lap.mergesp(self.fixed_diag_indices, fixed_diag, diag_indices, diag_maps)

        return (diag_indices, diag_maps), (tril_indices, tril_maps)

    def create_with_new_edge_index(self, edge_index):
        new_builder = self.__class__(
            self.size, edge_index, self.d,
            normalised=self.normalised, deg_normalised=self.deg_normalised, add_hp=self.add_hp, add_lp=self.add_lp,
            augmented=self.augmented)
        new_builder.train(self.training)
        return new_builder


class DiagLaplacianBuilder(LaplacianBuilder):
    """Learns a a Sheaf Laplacian with diagonal restriction maps"""

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, add_hp=False, add_lp=False,
                 augmented=True):
        super(DiagLaplacianBuilder, self).__init__(
            size, edge_index, d, normalised, deg_normalised, add_hp, add_lp, augmented)

        self.diag_indices, self.tril_indices = lap.compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def normalise(self, diag, tril, row, col):
        if self.normalised:
            d_sqrt_inv = (diag + 1).pow(-0.5) if self.augmented else diag.pow(-0.5)
            left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = d_sqrt_inv * diag * d_sqrt_inv
        elif self.deg_normalised:
            deg_sqrt_inv = (self.deg + 1).pow(-0.5) if self.augmented else self.deg.pow(-0.5)
            deg_sqrt_inv = deg_sqrt_inv.unsqueeze(-1)
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float('inf'), 0)
            left_norm, right_norm = deg_sqrt_inv[row], deg_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = deg_sqrt_inv * diag * deg_sqrt_inv
        return diag, tril

    def forward(self, maps):
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        row, _ = self.edge_index

        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.size)

        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        tril_maps, diag_maps = tril_maps.view(-1), diag_maps.view(-1)

        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        edge_index, weights = lap.mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        return (edge_index, weights), saved_tril_maps


class DiagHodgeLaplacianBuilder(nn.Module):
    """Build a diagonal sheaf Hodge term from signed simplex incidences.

    Each incidence owns a learned diagonal restriction map. Incidences that
    share a node (lower term) or a face (upper term) are paired to form an
    operator on edge stalks.
    """

    def __init__(self, edge_size, incidence_index, incidence_sign, d,
                 edge_dim, normalised=False, deg_normalised=False,
                 augmented=True):
        super(DiagHodgeLaplacianBuilder, self).__init__()
        self.edge_size = edge_size
        self.d = d
        self.edge_dim = edge_dim
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.augmented = augmented

        self.register_buffer('incidence_index', incidence_index.long())
        self.register_buffer('incidence_sign', incidence_sign)
        edge_index = incidence_index[edge_dim].long()
        shared_index = incidence_index[1 - edge_dim].long()
        self.register_buffer('edge_index', edge_index)

        pair_left, pair_right = self._incidence_pairs(shared_index)
        self.register_buffer('pair_left', pair_left)
        self.register_buffer('pair_right', pair_right)

        channels = torch.arange(d, device=incidence_index.device)
        if pair_left.numel() == 0:
            matrix_index = torch.empty(
                (2, 0), dtype=torch.long, device=incidence_index.device)
        else:
            row_edges = edge_index[pair_left]
            col_edges = edge_index[pair_right]
            rows = row_edges.unsqueeze(1) * d + channels.unsqueeze(0)
            cols = col_edges.unsqueeze(1) * d + channels.unsqueeze(0)
            matrix_index = torch.stack(
                (rows.reshape(-1), cols.reshape(-1)), dim=0)
        self.register_buffer('matrix_index', matrix_index)

        degree_values = torch.ones(
            edge_index.numel(), dtype=torch.float, device=edge_index.device)
        incidence_degree = scatter_add(
            degree_values, edge_index, dim=0, dim_size=edge_size)
        self.register_buffer('incidence_degree', incidence_degree)

    @staticmethod
    def _incidence_pairs(shared_index):
        if shared_index.numel() == 0:
            empty = torch.empty(
                (0,), dtype=torch.long, device=shared_index.device)
            return empty, empty

        order = torch.argsort(shared_index)
        sorted_shared = shared_index[order]
        _, counts = torch.unique_consecutive(sorted_shared, return_counts=True)
        groups = torch.split(order, counts.tolist())
        pair_left = torch.cat([
            group.repeat_interleave(group.numel()) for group in groups])
        pair_right = torch.cat([
            group.repeat(group.numel()) for group in groups])
        return pair_left, pair_right

    def forward(self, maps):
        if self.pair_left.numel() == 0:
            empty_values = maps.new_empty((0,))
            return self.matrix_index, empty_values

        signs = self.incidence_sign.to(dtype=maps.dtype).unsqueeze(-1)
        signed_maps = signs * maps
        pair_values = (
            signed_maps[self.pair_left] * signed_maps[self.pair_right])

        if self.normalised:
            diagonal = scatter_add(
                maps.square(), self.edge_index, dim=0,
                dim_size=self.edge_size)
            diagonal_inv_sqrt = (
                diagonal + int(self.augmented)).pow(-0.5)
            diagonal_inv_sqrt.masked_fill_(
                ~torch.isfinite(diagonal_inv_sqrt), 0)
            row_edges = self.edge_index[self.pair_left]
            col_edges = self.edge_index[self.pair_right]
            pair_values = (
                diagonal_inv_sqrt[row_edges]
                * pair_values
                * diagonal_inv_sqrt[col_edges])
        elif self.deg_normalised:
            degree_inv_sqrt = (
                self.incidence_degree.to(dtype=maps.dtype)
                + int(self.augmented)).pow(-0.5)
            degree_inv_sqrt.masked_fill_(
                ~torch.isfinite(degree_inv_sqrt), 0)
            row_edges = self.edge_index[self.pair_left]
            col_edges = self.edge_index[self.pair_right]
            pair_values = (
                degree_inv_sqrt[row_edges].unsqueeze(-1)
                * pair_values
                * degree_inv_sqrt[col_edges].unsqueeze(-1))

        sparse = torch.sparse_coo_tensor(
            self.matrix_index,
            pair_values.reshape(-1),
            size=(self.edge_size * self.d, self.edge_size * self.d),
            device=maps.device,
            dtype=maps.dtype,
        ).coalesce()
        return sparse.indices(), sparse.values()
