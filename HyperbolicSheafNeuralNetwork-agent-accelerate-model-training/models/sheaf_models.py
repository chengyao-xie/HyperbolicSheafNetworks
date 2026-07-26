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

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)

        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='leaky_relu')

        if sheaf_act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    # LocalConcatSheafLearner 的前向计算。
    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])
