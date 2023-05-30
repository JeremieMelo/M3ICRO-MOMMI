from typing import Optional
import numpy as np
import torch

from pyutils.compute import gen_gaussian_noise
from pyutils.general import logger
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import Tensor, nn
from torch.types import Device

from .utils import pad_quantize_fn
from pyutils.general import print_stat

__all__ = ["DPE_Layer_BASE"]


class DPE_Layer_BASE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_pads: int = 5,
        mini_block: int = 5,
        w_bit: int = 16,
        in_bit: int = 16,
        # constant scaling factor from intensity to detected voltages
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe: Optional[nn.Module] = None,
        pad_max: float = 1.0,
        mode: str = "usv",
        path_multiplier: int = 2,
        unfolding: bool = True,
        sigma_trainable: str = "row_col",
        device: Device = torch.device("cuda"),
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_pads = n_pads
        self.mini_block = mini_block
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.pad_max = pad_max
        # constant scaling factor from intensity to detected voltages
        self.input_uncertainty = input_uncertainty
        self.pad_noise_std = pad_noise_std
        self.mode = mode
        self.path_multiplier = path_multiplier
        self.unfolding = unfolding
        self.sigma_trainable = sigma_trainable

        # data dpe
        self.dpe = dpe

        self.verbose = verbose
        self.device = device

        # allocate parameters
        self.weight = None
        self.sigma = None
        self.x_zero_pad = None

        # quantization tool
        self.pad_quantizer = pad_quantize_fn(max(2, self.w_bit), v_max=pad_max, quant_ratio=1)
        self.sigma_quantizer = weight_quantize_fn(w_bit=8, alg="dorefa_sym")
        self.input_quantizer = input_quantize_fn(in_bit=in_bit, device=device, alg="normal")

        self._requires_grad_dpe = True
        self.input_er = 0
        self.input_max = 6
        self.input_snr = 0
        self.detection_snr = 0
        self.pad_noise_std = 0
        self.u, _, self.v = torch.randn(1, 1, mini_block, mini_block, dtype=torch.cfloat, device=device).svd()
        self.uv = torch.stack([self.u, self.v])

    def build_parameters(self, bias: bool):
        raise NotImplementedError

    def reset_parameters(self, fan_in=None):
        # nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.mode not in {"bsb", "fsf"}:
            if self.w_bit < 2:  # not trainable, all set to 0
                self.weight.data.fill_(-100)
                self.weight.requires_grad_(False)
            else:
                nn.init.uniform_(self.weight, -3, 3)  # compatible with weight quantizer

        if hasattr(self, "in_channels_flat"):
            fan_in = self.in_channels_flat
        else:
            fan_in = self.in_channels
        W = torch.randn(
            self.grid_dim_y, self.grid_dim_x, self.mini_block, self.mini_block, dtype=torch.cfloat, device=self.device
        ).mul((1 / fan_in) ** 0.5)

        S = torch.linalg.svdvals(W)
        if self.sigma is not None:
            self.sigma.data[0, ..., 0].copy_(S)
            self.sigma.data[1:, ..., 0].fill_(1)
            self.sigma.data[..., 1].fill_(0)

        if self.bias is not None:
            nn.init.uniform_(self.bias, 0, 0)

    def requires_grad_dpe(self, mode: bool = True):
        self._requires_grad_dpe = mode

    def set_input_er(self, er: float = 0, x_max: float = 6.0) -> None:
        ## extinction ratio of input modulator
        self.input_er = er
        self.input_max = x_max

    def set_input_snr(self, snr: float = 0) -> None:
        self.input_snr = snr

    def set_detection_snr(self, snr: float = 0) -> None:
        self.detection_snr = snr

    def add_input_noise(self, x: Tensor) -> Tensor:
        if 1e-5 < self.input_er < 80:
            x_max = self.input_max
            x_min = x_max / 10 ** (self.input_er / 10)
            x = x.mul((x_max - x_min) / x_max).add(x_min)
        if 1e-5 < self.input_snr < 80:
            avg_noise_power = 1 / 10 ** (self.input_snr / 10)
            noise = gen_gaussian_noise(x, 1, noise_std=avg_noise_power**0.5)
            return x.mul(noise)
        return x

    def add_detection_noise(self, x: Tensor) -> Tensor:
        if 1e-5 < self.detection_snr < 80:
            avg_noise_power = 0.5 / 10 ** (self.detection_snr / 10)
            noise = gen_gaussian_noise(x, 0, noise_std=avg_noise_power**0.5)
            return x.add(noise)
        return x

    @property
    def _weight(self):
        # control pads to complex transfer matrix
        # [p, q, n_pads] real -> [p, q, k, k] complex

        if self.sigma is not None:
            if self.sigma.dim() == 5:
                sigma = torch.view_as_complex(self.sigma)
            else:
                sigma = self.sigma
            if not torch.is_complex(sigma):
                sigma = self.sigma_quantizer(sigma)

        weight = self.pad_quantizer(self.weight)
        batch = tuple(weight.shape[0:-1])
        out_shape = batch + (self.mini_block, self.mini_block)

        weight = self.dpe(weight.flatten(0, -2), self._requires_grad_dpe)
        weight = weight.view(out_shape)

        if weight.dim() == 4:
            if self.mode == "u":
                pass
            elif self.mode == "usu":
                weight = weight.matmul(weight.mul(sigma[0].unsqueeze(-1)))  # U S U
            elif self.mode == "su":
                weight = weight.mul(sigma[0].unsqueeze(-1))  # S U
            elif self.mode == "us":
                weight = weight.mul(sigma[0].unsqueeze(-2))  # U S
            else:
                raise NotImplementedError
        elif weight.dim() == 5:
            if len(self.mode) >= 3:
                _weight = weight[0]  # v
                s_count = 0
                u_count = 1
                p_count = 0
                for m in self.mode[-2::-1]:  # "sususu..."
                    if m in {"u", "v"}:
                        _weight = weight[u_count].matmul(_weight)
                        u_count += 1
                    elif m == "s":
                        _weight = _weight.mul(sigma[s_count].unsqueeze(-1))
                        s_count += 1
                    elif m == "p":
                        _weight = _weight.mul(torch.exp(1j * self.phases[p_count].unsqueeze(-1)))
                        p_count += 1
                weight = _weight
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return weight  # [p, q, k, k] complex

    def _forward_impl(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self._forward_impl(x)
