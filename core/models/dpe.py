"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-05-25 00:45:19
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-29 18:48:33
"""

from typing import List

from .dpe_base import DPE_BASE

__all__ = ["DPE"]


class DPE(DPE_BASE):
    """Differentiable photonic device estimator (DPE)"""

    def __init__(
        self,
        n_ports: int,
        n_pads: int,
        *args,
        act_cfg: dict = dict(type="ReLU", inplace=True),
        hidden_dims: List[int] = [256, 256, 128, 128, 128],
        dropout: float = 0.0,
        is_complex: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            n_ports=n_ports,
            n_pads=n_pads,
            *args,
            act_cfg=act_cfg,
            hidden_dims=hidden_dims,
            dropout=dropout,
            is_complex=is_complex,
            **kwargs,
        )
