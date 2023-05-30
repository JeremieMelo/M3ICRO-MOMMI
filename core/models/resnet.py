"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:24:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:24:50
"""
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.activation import build_activation_layer
from pyutils.general import logger, print_stat
from torch import Tensor, nn
from torch.types import Device, _size

from .dpe_nn_base import DPE_NN_BASE

for handler in logger.root.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setLevel(logging.INFO)
__all__ = [
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]


def conv3x3(
    in_channels: int,
    out_channel: int,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    dilation: Union[int, _size] = 1,
    groups: int = 1,
):
    conv = torch.nn.Conv2d(
        in_channels,
        out_channel,
        3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    return conv


def conv1x1(
    in_channels: int,
    out_channel: int,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    dilation: Union[int, _size] = 1,
    groups: int = 1,
):
    conv = nn.Conv2d(
        in_channels,
        out_channel,
        1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    return conv


def Linear(
    in_channels: int,
    out_channels: int,
    bias: bool = False,
):
    linear = nn.Linear(in_channels, out_channels, bias=bias)
    return linear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        act_cfg: dict = dict(type="ReLU", inplace=True),
    ) -> None:
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = conv3x3(
            in_planes,
            planes,
            bias=False,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        if act_cfg is not None:
            self.act1 = build_activation_layer(act_cfg)
        else:
            self.act1 = None
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.conv2 = conv3x3(
            planes,
            planes,
            bias=False,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if act_cfg is not None:
            self.act2 = build_activation_layer(act_cfg)
        else:
            self.act2 = None

        self.shortcut = nn.Identity()
        # self.shortcut.conv1_spatial_sparsity = self.conv1.bp_input_sampler.spatial_sparsity
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    bias=False,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        # unique parameters
        act_cfg: dict = dict(type="ReLU", inplace=True),
    ) -> None:
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = conv1x1(
            in_planes,
            planes,
            bias=False,
            stride=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        if act_cfg is not None:
            self.act1 = build_activation_layer(act_cfg)
        else:
            self.act1 = None
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv3x3(
            planes,
            planes,
            bias=False,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if act_cfg is not None:
            self.act2 = build_activation_layer(act_cfg)
        else:
            self.act2 = None
        # self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.conv3 = conv1x1(
            planes,
            self.expansion * planes,
            bias=False,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        if act_cfg is not None:
            self.act3 = build_activation_layer(act_cfg)
        else:
            self.act3 = None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    bias=False,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(nn.Module):
    """MZI ResNet (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication."""

    def __init__(
        self,
        block,
        num_blocks,
        in_planes: int,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_classes: int,
        norm: str = "bn",
        act_cfg: dict = dict(type="ReLU", inplace=True),
        bias: bool = False,
        device: Device = torch.device("cuda"),
        verbose: bool = False,
    ) -> None:
        super().__init__()

        # resnet params
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = in_planes  # 64
        self.img_height = img_height
        self.img_width = img_width

        self.in_channels = in_channels
        self.num_classes = num_classes

        # list of block size
        self.norm = None if norm.lower() == "none" else norm
        self.act_cfg = act_cfg

        self.bias = bias

        self.device = device

        # build layers
        blkIdx = 0
        self.conv1 = conv3x3(
            in_channels,
            in_planes,
            bias=False,
            stride=1 if img_height <= 64 else 2,  # downsample for imagenet, dogs, cars
            padding=1,
        )
        # self.conv1 = nn.Conv2d(in_channels, in_planes, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        blkIdx += 1

        self.layer1 = self._make_layer(
            block,
            in_planes,
            num_blocks[0],
            stride=1,
            act_cfg=act_cfg,
        )
        blkIdx += 1

        self.layer2 = self._make_layer(
            block,
            in_planes * 2,
            num_blocks[1],
            stride=2,
            act_cfg=act_cfg,
        )
        blkIdx += 1

        self.layer3 = self._make_layer(
            block,
            in_planes * 4,
            num_blocks[2],
            stride=2,
            act_cfg=act_cfg,
        )
        blkIdx += 1

        self.layer4 = self._make_layer(
            block,
            in_planes * 8,
            num_blocks[3],
            stride=2,
            act_cfg=act_cfg,
        )
        blkIdx += 1

        n_channel = in_planes * 8 if num_blocks[3] > 0 else in_planes * 4
        self.linear = nn.Sequential(
            Linear(
                n_channel * block.expansion,
                self.num_classes,
                bias=False,
            ),
        )

    def reset_parameters(self):
        """the weights of conv layer and fully connected layers
        are both initilized with Xavier algorithm, In particular,
        we set the parameters to random values uniformly drawn from [-a, a]
        where a = sqrt(6 * (din + dout)), for batch normalization
        layers, y=1, b=0, all bias initialized to 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        act_cfg: dict = dict(type="ReLU", inplace=True),
    ):
        if num_blocks == 0:
            return nn.Identity()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_parameter_groups(self, weight_decay: float = 0, lr=None):
        param_optimizer = set(self.parameters())
        no_decay_parameters = []
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d)):
                no_decay_parameters.append(m.weight)
                no_decay_parameters.append(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                if m.bias is not None:
                    no_decay_parameters.append(m.bias)
        decay_parameters = list(param_optimizer - set(no_decay_parameters))

        optimizer_grouped_parameters = [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            },  # no decay for finetuning/distill
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]
        return optimizer_grouped_parameters

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if x.size(-1) > 64:  # 224 x 224, e.g., cars, dogs, imagenet
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, *args, **kwargs)


def ResNet20(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3, 0], 16, *args, **kwargs)


def ResNet32(*args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5, 0], 16, *args, **kwargs)


def ResNet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], 64, *args, **kwargs)


def ResNet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], 64, *args, **kwargs)


def ResNet101(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], 64, *args, **kwargs)


def ResNet152(*args, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], 64, *args, **kwargs)
