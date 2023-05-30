"""MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Block", "mobilenet_v3", "hswish"]


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(
        self,
        in_size,
        reduction=4,
        device=torch.device("cuda"),
    ):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
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
        device=torch.device("cuda"),
    ):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nonlinear1 = nonlinear
        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nonlinear2 = nonlinear

        self.stride = stride
        if semodule:
            self.se = SeModule(expand_size)
        else:
            self.se = None

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential(nn.Identity())
        self.use_res = stride == 1 and in_size == out_size
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         # nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
        #         AddDropMRRConv2d(
        #             in_size,
        #             out_size,
        #             kernel_size=1,
        #             stride=1,
        #             padding=0,
        #             bias=False,
        #             mode=mode,
        #             in_bit=in_bit,
        #             w_bit=w_bit,
        #             mrr_a=mrr_a,
        #             mrr_r=mrr_r,
        #             device=device,
        #         ),
        #         nn.BatchNorm2d(out_size),
        #     )

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


class MobileNetV3(nn.Module):
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
        img_height=32,
        img_width=32,
        device=torch.device("cuda"),
    ):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        if img_height <= 32 and img_width <= 32:  ## small dataset, e.g.,  CIFAR
            init_stride = 1
            self.cfg = self.cifar_cfg
        elif img_height <= 64 and img_width <= 64:  ##tinyimagenet
            init_stride = 1
            self.cfg = self.tinyimagenet_cfg
        else:
            init_stride = 2
            self.cfg = self.imagenet_cfg

        # building first layer
        assert img_height % 32 == 0
        last_channel = 1024
        width_mult = 1
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        features = [
            nn.Conv2d(3, 16, kernel_size=3, stride=init_stride, padding=1, bias=False),
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
                    device=device,
                )
            )
            input_channel = output_channel

        features += [
            nn.Conv2d(
                96,
                576,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(576),
            hswish(),
        ]

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(
                576,
                last_channel,
                bias=True,
            ),
            nn.BatchNorm1d(last_channel),
            hswish(),
            nn.Dropout(p=0.2),
            nn.Linear(
                last_channel,
                num_classes,
                bias=True,
            ),
        )

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

    def init_from_pretrained_model(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenet_v3(pretrained=False, **kwargs):
    return MobileNetV3(**kwargs)
