"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-26 00:36:10
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-02 19:24:43
"""

import logging
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from mmcv.cnn.bricks.activation import build_activation_layer
from multimethod import multimethod
from pyutils.compute import gen_gaussian_noise
from pyutils.general import logger
from pyutils.torch_train import set_torch_deterministic
from torch import is_complex, nn
from torch.functional import Tensor
from torch.types import Device

from core.models.utils import dpe_grad_estimator

logging.root.setLevel(logging.INFO)

__all__ = ["DPE_BASE"]


class DPE_BASE(nn.Module):
    _linear = (nn.Linear,)
    cfg = [128, 128, 128]

    def __init__(
        self,
        n_pads: int,
        n_ports: int,
        act_cfg: dict = dict(type="ReLU", inplace=True),
        hidden_dims: List[int] = [128, 128, 128],
        dropout: float = 0.0,
        is_complex: bool = True,
        is_symmetric: bool = True,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.n_pads = n_pads
        self.n_ports = n_ports
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.cfg = hidden_dims
        self.cfg.insert(0, n_pads * 2)
        self.is_complex = is_complex
        self.is_symmetric = is_symmetric
        if is_complex:
            self.cfg.append(n_ports * n_ports * 2)
        else:
            self.cfg.append(n_ports * n_ports)  # predict all
        self.build_layers()
        self._requires_grad = True
        self.device = device
        self._lookup = None
        self.set_pad_noise(0)
        self.set_dpe_noise_ratio(0)
        self.dpe_grad_estimator = dpe_grad_estimator(self)

    def build_layers(self):
        self.layers = []
        self.freq = nn.Parameter(torch.ones(self.n_pads) * np.pi)
        self.phase_bias = nn.Parameter(torch.zeros(self.n_pads))
        for in_channel, out_channel in zip(self.cfg[:-2], self.cfg[1:-1]):
            self.layers += [
                nn.Linear(in_channel, out_channel),
                build_activation_layer(self.act_cfg),
            ]
        if self.dropout > 0:
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(self.cfg[-2], self.cfg[-1]))
        self.layers = nn.Sequential(*self.layers)

    def set_pad_noise(self, noise_std: float = 0.0) -> None:
        self.pad_noise_std = noise_std

    def add_pad_noise(self, x: Tensor) -> Tensor:
        if self.pad_noise_std > 1e-5:
            x = x + gen_gaussian_noise(x, 0, self.pad_noise_std)
        return x

    def set_dpe_noise_ratio(self, ratio: float) -> None:
        self.dpe_noise_ratio = ratio
        assert 0 <= ratio <= 1

    def build_lookup_table(
        self,
        data: Tensor,
        targets: Tensor,
        w_bit: int = 4,  # how many levels, not typical bitwidth
        pad_max: float = 1.0,
    ) -> Callable:
        # data [bs, n] # n control pads permittivities
        # targets [bs, 50] # flattened complex 5x5 transfer matrix or [bs, 16] # flattened real 4x4 transfer matrix
        pad_level = data.data.div(pad_max / (w_bit - 1)).float()  # [bs, n_pads]
        # encode the level to digits using weighted sum, so we can sort
        coeff = w_bit ** torch.arange(data.shape[-1] - 1, -1, -1, device=data.device).float()  # [n_pads]
        weights = pad_level.matmul(coeff.unsqueeze(1)).squeeze(1)  # [bs, n]x[n,1] -> [bs]
        indices = weights.argsort()
        # sort the targets as an ordered table
        targets = targets[indices].view([w_bit] * data.shape[-1] + list(targets.shape[1:])).to(self.device)

        def _lookup(x: Tensor) -> Tensor:
            ## x [bs, n_pads]
            x = x.data.div(pad_max / (w_bit - 1)).round_().long()
            indices = tuple([i.squeeze(1) for i in x.chunk(x.shape[1], dim=1)])
            return targets[indices]  # [bs, ...]

        self._lookup = _lookup
        return _lookup

    def requires_grad(self, flag: bool = True) -> None:
        self._requires_grad = flag
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.requires_grad_(flag)
                if m.bias is not None:
                    m.bias.requires_grad_(flag)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.requires_grad_(flag)
                m.bias.requires_grad_(flag)
            elif isinstance(m, (torch.nn.PReLU,)):
                m.weight.requires_grad_(flag)

    def _preprocess(self, x: Tensor) -> Tensor:
        x = torch.cat([x.mul(self.freq).add(self.phase_bias).cos(), x], 1)
        return x

    def _postprocess(self, x: Tensor) -> Tensor:
        if self.is_symmetric:
            if self.is_complex:
                x = x.reshape(-1, self.n_ports, self.n_ports, 2)
            else:
                x = x.reshape(-1, self.n_ports, self.n_ports)
            x = (x + x.transpose(1, 2)) / 2
        return x

    def forward(self, x: Tensor, differentiable: bool = True) -> Tensor:
        if differentiable:
            x = self.add_pad_noise(x)
            with torch.cuda.amp.autocast():
                W = self._postprocess(self.layers(self._preprocess(x)))
            W = W.float()
            if self.is_complex:
                W = torch.view_as_complex(W.reshape(W.shape[0], self.n_ports, self.n_ports, 2))
            else:
                W = W.reshape(W.shape[0], self.n_ports, self.n_ports)

            if 1e-5 < self.dpe_noise_ratio < 1 - 1e-5:
                W = W.mul(1 - self.dpe_noise_ratio) + self.dpe_grad_estimator(x, W).mul(self.dpe_noise_ratio)
            elif self.dpe_noise_ratio >= 1 - 1e-5:
                W = self.dpe_grad_estimator(x, W)
            return W
        else:
            if self._lookup is None:
                logger.error("No lookup table found. Please call build_lookup_table() first")
                raise ValueError
            x = self._lookup(x)
            return x
