"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-05-25 22:19:52
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-05-26 01:02:27
"""
from typing import Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.types import Device
import torch.nn.functional as F

from core.models.butterfly_utils import BatchTrainableButterfly
from core.models.layers.dpe_layer_base import DPE_Layer_BASE
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

__all__ = ["DPEConv2d"]


class _DPEConv2dMultiPath(DPE_Layer_BASE):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        n_pads: int = 5,
        mini_block: int = 5,
        bias: bool = False,
        w_bit: int = 16,
        in_bit: int = 16,
        # constant scaling factor from intensity to detected voltages
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe=None,
        pad_max: float = 1.0,
        sigma_trainable: str = "row_col",
        mode: str = "usv",
        path_multiplier: int = 2,
        unfolding: bool = True,
        device: Device = torch.device("cuda"),
        verbose: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_pads=n_pads,
            mini_block=mini_block,
            w_bit=w_bit,
            in_bit=in_bit,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            mode=mode,
            sigma_trainable=sigma_trainable,
            device=device,
            verbose=verbose,
        )
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        # allocate parameters
        self.weight = None
        self.x_zero_pad = None
        self.sigma_trainable = sigma_trainable
        self.path_multiplier = path_multiplier
        self.unfolding = unfolding

        self.build_parameters(bias=bias)

    def build_parameters(self, bias: bool) -> None:
        self.in_channels_flat = self.in_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups
        self.in_channels_pad = int(np.ceil(self.in_channels_flat / self.mini_block).item() * self.mini_block)

        if self.unfolding:
            self.out_channels_pad = (
                int(np.ceil(self.out_channels / 2 / self.mini_block).item() * self.mini_block) * self.path_multiplier
            )
        else:
            self.out_channels_pad = (
                int(np.ceil(self.out_channels / self.mini_block).item() * self.mini_block) * self.path_multiplier
            )
        self.grid_dim_y = self.out_channels_pad // self.mini_block
        self.grid_dim_x = self.in_channels_pad // self.mini_block
        # each block has n_pads real parameters
        if self.mode == "fsf":
            self.weight = None
            eye = torch.eye(self.mini_block, self.mini_block, dtype=torch.cfloat, device=self.device)
            self.b = torch.fft.fft(eye, norm="ortho")
            self.p = torch.fft.ifft(eye, norm="ortho")  # [p/2, q, k, k]

        elif self.mode == "bsb":
            self.weight = nn.ModuleList(
                [
                    BatchTrainableButterfly(
                        batch_size=(self.grid_dim_y, self.grid_dim_x),
                        length=self.mini_block,
                        device=self.device,
                        bit_reversal=False,
                    ),
                    BatchTrainableButterfly(
                        batch_size=(self.grid_dim_y, self.grid_dim_x),
                        length=self.mini_block,
                        device=self.device,
                        mode="full_reverse",
                        bit_reversal=False,
                    ),
                ]
            )
        elif self.mode in {"u", "us", "su", "usu"}:
            self.weight = nn.Parameter(torch.empty(self.grid_dim_y, self.grid_dim_x, self.n_pads, device=self.device))
        elif len(self.mode) >= 3:
            num_mmis = self.mode.count("u") + self.mode.count("v")
            self.weight = nn.Parameter(
                torch.empty(num_mmis, self.grid_dim_y, self.grid_dim_x, self.n_pads, device=self.device)
            )
        else:
            raise NotImplementedError

        self.sigma = None
        num_sigma = self.mode.count("s")
        if self.sigma_trainable.startswith("row_col"):
            self.sigma = nn.Parameter(
                torch.ones(num_sigma, self.grid_dim_y, self.grid_dim_x, self.mini_block, 2, device=self.device)
                / self.mini_block
            )
        elif self.sigma_trainable.startswith("row"):
            self.sigma = nn.Parameter(num_sigma, torch.ones(self.grid_dim_y, 1, self.mini_block, 2, device=self.device))
        elif self.sigma_trainable.startswith("col"):
            self.sigma = nn.Parameter(num_sigma, torch.ones(1, self.grid_dim_x, self.mini_block, 2, device=self.device))
        else:
            raise NotImplementedError(f"{self.sigma_trainable} is not supported")

        if self.sigma_trainable.endswith("_real"):
            self.sigma = nn.Parameter(self.sigma.data[..., 0])

        num_phi = self.mode.count("p")
        if self.sigma_trainable.startswith("row_col"):
            self.phases = nn.Parameter(
                torch.zeros(num_phi, self.grid_dim_y, self.grid_dim_x, self.mini_block, device=self.device)
                / self.mini_block
            )
        elif self.sigma_trainable.startswith("row"):
            self.phases = nn.Parameter(num_phi, torch.zeros(self.grid_dim_y, 1, self.mini_block, device=self.device))
        elif self.sigma_trainable.startswith("col"):
            self.phases = nn.Parameter(num_phi, torch.zeros(1, self.grid_dim_x, self.mini_block, device=self.device))
        else:
            raise NotImplementedError(f"{self.sigma_trainable} is not supported")

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

    def build_weight_unroll(self, weight: Tensor) -> Tensor:
        weight = weight.reshape(
            [
                self.path_multiplier,
                weight.shape[0] // self.path_multiplier,
                weight.shape[1],
                weight.shape[2],
                weight.shape[3],
            ]
        ).mean(0)
        # weight_p, weight_n = weight.chunk(2, dim=0)
        # weight = weight_p + weight_n # [p/2, q, k, k] complex
        weight = (
            weight.permute(0, 2, 1, 3)
            .reshape(self.out_channels_pad // self.path_multiplier, -1)[
                : self.out_channels // 2, : self.in_channels_flat
            ]
            .view(self.out_channels // 2, self.in_channels // self.groups, *self.kernel_size)
        )  # [outc/2, inc, kh, kw] complex
        weight = torch.view_as_real(weight)  # [outc/2, inc, kh, kw, 2] real
        weight = weight.permute(0, 4, 1, 2, 3).flatten(0, 1)

        return weight

    def build_weight(self, weight: Tensor) -> Tensor:
        weight = weight.reshape(
            [
                self.path_multiplier,
                weight.shape[0] // self.path_multiplier,
                weight.shape[1],
                weight.shape[2],
                weight.shape[3],
            ]
        ).mean(0)
        # weight_p, weight_n = weight.chunk(2, dim=0)
        # weight = weight_p + weight_n # [p/2, q, k, k] complex
        weight = (
            weight.permute(0, 2, 1, 3)
            .reshape(self.out_channels_pad // self.path_multiplier, -1)[: self.out_channels, : self.in_channels_flat]
            .view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        )  # complex weights [outc, inc, k, k]

        return weight

    def _build_weight(self):
        if self.mode == "bsb":
            weight = (
                self.weight[1]
                .build_weight()
                .matmul(self.weight[0].build_weight().mul(torch.view_as_complex(self.sigma[0]).unsqueeze(-1)))
            )
        elif self.mode == "fsf":
            weight = self.p.matmul(
                torch.view_as_complex(self.sigma[0]).unsqueeze(-1).mul(self.b)
            )  # [p, q, k, k] complex
            # print(weight.shape, self.grid_dim_y, self.grid_dim_y)
            # exit(0)
        else:
            weight = self._weight  # [p, q, k, k] complex
        return weight

    def _forward_impl(self, x: Tensor) -> Tensor:
        # assert (
        #     x.size(-1) == self.in_channels
        # ), f"[E] Input dimension does not match the weight size {self.out_channels, self.in_channels}, but got input size ({tuple(x.size())}))"

        # modulation
        # x: [bs, inc, h, w]
        x = self.add_input_noise(x)

        weight = self._build_weight()

        if self.unfolding:
            weight = self.build_weight_unroll(weight)  # real
        else:
            weight = self.build_weight(weight)  # complex
        if torch.is_complex(weight):
            x = torch.complex(
                F.conv2d(
                    x,
                    weight.real,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                ),
                F.conv2d(
                    x,
                    weight.imag,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                ),
            ).abs()
        else:
            x = F.conv2d(
                x,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        x = self.add_detection_noise(x)

        if self.bias is not None:
            x = x + self.bias[None, :, None, None]
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class DPEConv2d(DPE_Layer_BASE):
    _conv_types = _DPEConv2dMultiPath

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        n_pads: int = 5,
        mini_block: int = 5,
        bias: bool = True,
        w_bit: int = 16,
        in_bit: int = 16,
        # constant scaling factor from intensity to detected voltages
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe=None,
        pad_max: float = 1.0,
        sigma_trainable: str = "row_col",
        mode: str = "usv",
        path_multiplier: int = 2,
        unfolding: bool = True,
        device: Device = torch.device("cuda"),
        verbose: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_pads=n_pads,
            mini_block=mini_block,
            w_bit=w_bit,
            in_bit=in_bit,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            mode=mode,
            path_multiplier=path_multiplier,
            unfolding=unfolding,
            device=device,
            verbose=verbose,
        )

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.in_channels_pos = self.in_channels
        self.in_channels_neg = 0 if unfolding else self.in_channels
        self._conv_pos = _DPEConv2dMultiPath(
            in_channels=self.in_channels_pos,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            n_pads=n_pads,
            mini_block=mini_block,
            bias=False,
            w_bit=w_bit,
            in_bit=in_bit,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            sigma_trainable=sigma_trainable,
            mode=mode,
            path_multiplier=path_multiplier,
            unfolding=unfolding,
            device=device,
            verbose=verbose,
        )
        self._conv_neg = (
            _DPEConv2dMultiPath(
                in_channels=self.in_channels_pos,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                n_pads=n_pads,
                mini_block=mini_block,
                bias=False,
                w_bit=w_bit,
                in_bit=in_bit,
                input_uncertainty=input_uncertainty,
                pad_noise_std=pad_noise_std,
                dpe=dpe,
                pad_max=pad_max,
                sigma_trainable=sigma_trainable,
                mode=mode,
                path_multiplier=path_multiplier,
                unfolding=unfolding,
                device=device,
                verbose=verbose,
            )
            if self.in_channels_neg > 0
            else None
        )

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.reset_parameters(fan_in=self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if self.bias is not None:
            self.bias.data.zero_()

    def requires_grad_dpe(self, mode: bool = True):
        self._requires_grad_dpe = mode
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.requires_grad_dpe(mode)

    def set_input_er(self, er: float = 0, x_max: float = 6.0) -> None:
        ## extinction ratio of input modulator
        self.input_er = er
        self.input_max = x_max
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_input_er(er, x_max)

    def set_input_snr(self, snr: float = 0) -> None:
        self.input_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_input_snr(snr)

    def set_detection_snr(self, snr: float = 0) -> None:
        self.detection_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_detection_snr(snr)

    @property
    def _weight(self):
        # control pads to complex transfer matrix
        # [p, q, n_pads] real -> [p, q, k, k] complex
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m._build_weight())
        return weights

    @property
    def _weight_unroll(self):
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m.build_weight_unroll(m._build_weight()))
        return weights

    @property
    def _weight_complex(self):
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m.build_weight(m._build_weight()))
        return weights

    def _forward_impl(self, x):
        y = self._conv_pos(x)
        if self._conv_neg is not None:
            y_neg = self._conv_neg(x)
            y = y - y_neg

        if self.bias is not None:
            y = y + self.bias[None, :, None, None]
        return y

    def get_output_dim(self, img_height: int, img_width: int) -> Tuple[int, int]:
        h_out = (img_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        w_out = (img_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1
        return (int(h_out), int(w_out))

    def forward(self, x):
        if self.in_bit <= 8:
            x = self.input_quantizer(x)
        return self._forward_impl(x)
