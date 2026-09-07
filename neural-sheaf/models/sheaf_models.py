# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from abc import abstractmethod
from torch import nn


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""
    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh",
                 edge_channels: int = 0, edge_scale_bands: int = 0,
                 adaptive_edge_scale_gates: bool = False,
                 adaptive_edge_scale_strength: float = 0.1):
        super(LocalConcatSheafLearner, self).__init__()
        self.out_shape = out_shape
        self.edge_channels = edge_channels
        self.edge_scale_bands = edge_scale_bands
        self.adaptive_edge_scale_gates = adaptive_edge_scale_gates
        self.adaptive_edge_scale_strength = adaptive_edge_scale_strength
        self.last_edge_scale_gates = None
        if edge_scale_bands:
            self.edge_scale_logits = nn.Parameter(torch.zeros(edge_scale_bands))
            if adaptive_edge_scale_gates:
                gate_hidden = max(8, edge_channels * 2)
                with torch.random.fork_rng(devices=[]):
                    self.edge_scale_gate_network = nn.Sequential(
                        nn.Linear(edge_channels, gate_hidden),
                        nn.ELU(),
                        nn.Linear(gate_hidden, edge_scale_bands),
                    )
                nn.init.zeros_(self.edge_scale_gate_network[-1].weight)
                nn.init.zeros_(self.edge_scale_gate_network[-1].bias)
            else:
                self.edge_scale_gate_network = None
        else:
            self.register_parameter('edge_scale_logits', None)
            self.edge_scale_gate_network = None
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + edge_channels,
            int(np.prod(out_shape)),
            bias=False,
        )

        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='leaky_relu')

        if sheaf_act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        inputs = [x_row, x_col]
        if self.edge_channels:
            if edge_attr is None:
                edge_attr = x.new_zeros((row.numel(), self.edge_channels))
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
            if self.edge_scale_bands:
                if self.edge_scale_gate_network is not None:
                    adaptive_logits = self.edge_scale_gate_network(edge_attr)
                    gates = torch.softmax(
                        self.adaptive_edge_scale_strength * torch.tanh(adaptive_logits)
                        + self.edge_scale_logits.unsqueeze(0),
                        dim=-1,
                    ) * self.edge_scale_bands
                    self.last_edge_scale_gates = gates.detach().mean(dim=0)
                else:
                    gates = torch.softmax(self.edge_scale_logits, dim=0)
                    gates = gates * self.edge_scale_bands
                    self.last_edge_scale_gates = gates.detach()
                edge_attr = torch.cat([
                    edge_attr[:, :self.edge_scale_bands] * gates,
                    edge_attr[:, self.edge_scale_bands:] * gates,
                ], dim=1)
            inputs.append(edge_attr)
        maps = self.linear1(torch.cat(inputs, dim=1))
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalConcatIncidenceSheafLearner(SheafLearner):
    """Learn restriction maps from features on two incident simplex orders."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatIncidenceSheafLearner, self).__init__()
        self.out_shape = out_shape
        self.linear1 = nn.Linear(in_channels * 2, int(np.prod(out_shape)), bias=False)
        nn.init.kaiming_normal_(
            self.linear1.weight, mode='fan_out', nonlinearity='leaky_relu')

        if sheaf_act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu

    def forward(self, upper_x, lower_x, incidence_index):
        """Return one learned restriction map per incidence."""
        upper_index, lower_index = incidence_index
        upper_features = torch.index_select(upper_x, dim=0, index=upper_index)
        lower_features = torch.index_select(lower_x, dim=0, index=lower_index)
        maps = self.act(self.linear1(torch.cat(
            [upper_features, lower_features], dim=1)))

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        return maps.view(-1, self.out_shape[0])
