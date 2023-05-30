"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-08-26 02:39:03
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-12 01:59:00
"""


import numpy as np
import torch
from pyutils.general import logger as lg
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn

from core.models.butterfly_utils import BatchTrainableButterfly

from .layers import DPEConv2d, DPELinear

__all__ = ["DPE_NN_BASE"]


class DPE_NN_BASE(nn.Module):
    _conv_linear = (DPELinear, DPEConv2d)
    _conv = (DPEConv2d,)
    _linear = (DPELinear,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def requires_grad_dpe(self, mode: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.requires_grad_dpe(mode)

    def set_input_er(self, er: float = 0.0, x_max: float = 6.0) -> None:
        self.input_er = er
        self.input_max = x_max
        first_layer = True
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                if first_layer:
                    m.set_input_er(er, 1)
                    first_layer = False
                else:
                    m.set_input_er(er, x_max)

    def set_input_snr(self, snr: float = 0.0) -> None:
        self.input_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.set_input_snr(snr)

    def set_detection_snr(self, snr: float = 0.0) -> None:
        self.input_detection_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.set_detection_snr(snr)

    def set_pad_noise(self, noise_std: float = 0.0) -> None:
        self.pad_noise_std = noise_std
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.set_pad_noise(noise_std)

    def get_parameter_groups(self, weight_decay: float = 0, lr: float = 1e-3):
        param_optimizer = list(self.named_parameters())

        no_decay = ["bias", "weight"]
        no_decay_group = [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]
        no_decay_reduce_lr_group = []
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        no_decay_reduce_lr_group.append(p)
        no_decay_group = list(set(no_decay_group) - set(no_decay_reduce_lr_group))
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },  # no decay for finetuning/distill
            {"params": no_decay_group, "weight_decay": 0.0, "lr": lr},
            {"params": no_decay_reduce_lr_group, "weight_decay": 0.0, "lr": lr / 10},
        ]
        return optimizer_grouped_parameters

    def load_from_teacher(self, teacher):
        bn_modules_student = []
        conv_modules_student = []
        linear_modules_student = []

        bn_modules_teacher = []
        conv_modules_teacher = []
        linear_modules_teacher = []

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules_student.append(m)
            elif isinstance(m, self._conv):
                conv_modules_student.append(m)
            elif isinstance(m, self._linear):
                linear_modules_student.append(m)

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules_teacher.append(m)
            elif isinstance(m, nn.Conv2d):
                conv_modules_teacher.append(m)
            elif isinstance(m, nn.Linear):
                linear_modules_teacher.append(m)

        # map batch norm
        for bn_stu, bn_tea in zip(bn_modules_student, bn_modules_teacher):
            bn_stu.weight.data.copy_(bn_tea.weight.data)
            bn_stu.bias.data.copy_(bn_tea.bias.data)

        # map bias
        for conv_stu, conv_tea in zip(conv_modules_student, conv_modules_teacher):
            if conv_tea.bias is not None:
                conv_stu.bias.data.copy_(conv_tea.bias.data)

        for linear_stu, linear_tea in zip(linear_modules_student, linear_modules_teacher):
            if linear_tea.bias is not None:
                linear_stu.bias.data.copy_(linear_tea.bias.data)

        # map linear/cponv layers
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=1e-2,
        )
        N = 3000 if self.unfolding else 1000

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N, last_epoch=-1)
        target = torch.cat([m.weight.data.flatten() for m in conv_modules_teacher + linear_modules_teacher])

        def _build_weight_stu(conv_modules, linear_modules):
            modules = conv_modules + linear_modules
            weights = torch.cat([m._weight_unroll[0].flatten() for m in modules])
            return weights

        def _build_weight_stu_nonlinear(conv_modules, linear_modules):
            modules = conv_modules + linear_modules
            weights = torch.cat(
                [m._weight_complex[0].flatten().abs() - m._weight_complex[1].flatten().abs() for m in modules]
            )
            return weights

        if self.unfolding:
            _build_weight = _build_weight_stu
        else:
            _build_weight = _build_weight_stu_nonlinear
        for i in range(N):
            weights = _build_weight(conv_modules_student, linear_modules_student)
            loss = torch.nn.functional.mse_loss(weights, target, reduction="sum")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 1000 == 0 or i == N - 1:
                lg.info(f"Step: {i}, loss={loss.item():.4f}")

    def set_butterfly_noise(self, noise_std: float = 0.0) -> None:
        for m in self.modules():
            if isinstance(m, BatchTrainableButterfly):
                m.phase_noise_std = noise_std

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
