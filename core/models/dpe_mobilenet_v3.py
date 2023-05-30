"""MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device, _size

from core.models.layers.dpe_conv2d import DPEConv2d
from core.models.layers.dpe_linear import DPELinear

from .dpe_nn_base import DPE_NN_BASE
from .dpe_resnet import conv1x1

__all__ = ["DPE_mobilenet_v3"]


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


def Linear(
    in_channels: int,
    out_channel: int,
    bias: bool = False,
    **unique_parameters,
):
    # linear = nn.Linear(in_channel, out_channel)
    linear = DPELinear(
        in_channels,
        out_channel,
        bias=bias,
        **unique_parameters,
    )
    return linear


class SeModule(nn.Module):
    def __init__(
        self,
        in_size,
        reduction=4,
        **unique_parameters,
    ):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            conv1x1(in_size, in_size // reduction, stride=1, padding=0, bias=False, **unique_parameters),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            conv1x1(in_size // reduction, in_size, stride=1, padding=0, bias=False, **unique_parameters),
            hsigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self,
        kernel_size,
        in_size,
        expand_size,
        out_size,
        nonlinear,
        semodule,
        stride,
        **unique_parameters,
    ):
        super(Block, self).__init__()

        # self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = conv1x1(in_size, expand_size, stride=1, padding=0, bias=False, **unique_parameters)

        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nonlinear1 = nonlinear
        # self.conv2 = nn.Conv2d(
        #     expand_size,
        #     expand_size,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=kernel_size // 2,
        #     groups=expand_size,
        #     bias=False,
        # )
        self.conv2 = DPEConv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
            **unique_parameters,
        )

        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nonlinear2 = nonlinear

        self.stride = stride
        if semodule:
            self.se = SeModule(expand_size, **unique_parameters)
        else:
            self.se = None

        # self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = conv1x1(expand_size, out_size, stride=1, padding=0, bias=False, **unique_parameters)

        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential(nn.Identity())
        self.use_res = stride == 1 and in_size == out_size

    def forward(self, x):
        input = x
        x = self.nonlinear1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.se != None:
            x = self.se(x)
        x = self.nonlinear2(x)
        x = self.bn3(self.conv3(x))

        x = x + self.shortcut(input) if self.use_res else x
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np

    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class MobileNetV3(DPE_NN_BASE):
    cifar_cfg = [  # total 11 blocks
        # k, exp, c,  se,     nl,  s,
        [3, 16, 16, True, nn.ReLU(), 1],
        [3, 72, 24, False, nn.ReLU(), 2],
        [3, 88, 24, False, nn.ReLU(), 1],
        [5, 96, 40, True, hswish(), 2],
        [5, 240, 40, True, hswish(), 1],
        [5, 240, 40, True, hswish(), 1],
        [5, 120, 48, True, hswish(), 1],
        [5, 144, 48, True, hswish(), 1],
        [5, 288, 96, True, hswish(), 2],
        [5, 576, 96, True, hswish(), 1],
        [5, 576, 96, True, hswish(), 1],
    ]
    imagenet_cfg = [  # total 11 blocks
        # k, exp, c,  se,     nl,  s,
        [3, 16, 16, True, nn.ReLU(), 2],
        [3, 72, 24, False, nn.ReLU(), 2],
        [3, 88, 24, False, nn.ReLU(), 1],
        [5, 96, 40, True, hswish(), 2],
        [5, 240, 40, True, hswish(), 1],
        [5, 240, 40, True, hswish(), 1],
        [5, 120, 48, True, hswish(), 1],
        [5, 144, 48, True, hswish(), 1],
        [5, 288, 96, True, hswish(), 2],
        [5, 576, 96, True, hswish(), 1],
        [5, 576, 96, True, hswish(), 1],
    ]

    def __init__(
        self,
        num_classes=10,
        in_channels: int = 3,
        img_height=32,
        img_width=32,
        block_list: List[int] = [5, 5],
        in_bit: int = 32,
        w_bit: int = 32,
        norm: str = "bn",
        act_cfg: dict = dict(type="ReLU", inplace=True),
        bias: bool = False,
        # constant scaling factor from intensity to detected voltages
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe: Optional[nn.Module] = None,
        pad_max: float = 1.0,
        n_pads: int = 5,
        sigma_trainable: str = "row_col",
        mode: str = "usv",
        path_multiplier: int = 2,
        unfolding: bool = True,
        device: Device = torch.device("cuda"),
        verbose: bool = False,
    ):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        if img_height <= 32 and img_width <= 32:  ## small dataset, e.g.,  CIFAR
            init_stride = 1
            self.cfg = self.cifar_cfg
        elif img_height <= 64 and img_width <= 64:  ##tinyimagenet
            init_stride = 1
            self.cfg = self.tinyimagenet_cfg
        else:
            init_stride = 2
            self.cfg = self.imagenet_cfg

        # list of block size
        self.block_list = block_list
        self.norm = None if norm.lower() == "none" else norm
        self.act_cfg = act_cfg
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.pad_max = pad_max
        self.n_pads = n_pads
        self.sigma_trainable = sigma_trainable
        self.mode = mode
        self.path_multiplier = path_multiplier
        self.unfolding = unfolding
        # constant scaling factor from intensity to detected voltages
        self.input_uncertainty = input_uncertainty
        self.pad_noise_std = pad_noise_std
        self.bias = bias

        self.dpe = dpe

        self.device = device

        unique_parameters = dict(
            mini_block=self.block_list[0],
            n_pads=n_pads,
            w_bit=w_bit,
            in_bit=in_bit,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            device=device,
            sigma_trainable=sigma_trainable,
            mode=mode,
            path_multiplier=path_multiplier,
            unfolding=self.unfolding,
            verbose=verbose,
        )

        # building first layer
        assert img_height % 32 == 0
        last_channel = 1024
        width_mult = 1
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        features = [
            # nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=init_stride, padding=1, bias=False),
            DPEConv2d(
                self.in_channels, 16, kernel_size=3, stride=init_stride, padding=1, bias=False, **unique_parameters
            ),
            nn.BatchNorm2d(16),
            hswish(),
        ]
        input_channel = 16
        # building mobile blocks
        for k, exp, c, se, nl, s in self.cfg:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            features.append(
                Block(
                    kernel_size=k,
                    in_size=input_channel,
                    expand_size=exp_channel,
                    out_size=output_channel,
                    nonlinear=nl,
                    semodule=se,
                    stride=s,
                    **unique_parameters,
                )
            )
            input_channel = output_channel

        features += [
            # nn.Conv2d(
            #     96,
            #     576,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0,
            #     bias=False,
            # ),
            DPEConv2d(
                96,
                576,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                **unique_parameters,
            ),
            nn.BatchNorm2d(576),
            hswish(),
        ]

        self.features = nn.Sequential(*features)

        # self.linear = nn.Linear(1280, num_classes)
        self.classifier = nn.Sequential(
            # nn.Dropout(dropout_rate),
            Linear(
                576,
                last_channel,
                bias=True,
                **unique_parameters,
            ),
            nn.BatchNorm1d(last_channel),
            hswish(),
            nn.Dropout(p=0.2),
            Linear(
                last_channel,
                num_classes,
                bias=True,
                **unique_parameters,
            ),
        )
        # self.reset_parameters()

    def init_from_pretrained_model(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def DPE_mobilenet_v3(pretrained=False, **kwargs):
    return MobileNetV3(**kwargs)
