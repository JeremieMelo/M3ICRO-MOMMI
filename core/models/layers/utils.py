"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-29 22:34:17
"""

import torch
from pyutils.general import logger

__all__ = ["pad_quantize_fn"]


def uniform_quantize(num_levels, gradient_clip=False):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # n = float(2 ** k - 1)
            n = num_levels - 1  # explicit assign number of quantization level,e.g., k=5 or 8
            out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return qfn().apply


class pad_quantize_fn(torch.nn.Module):
    def __init__(self, w_bit, quant_ratio: float = 1.0, v_max: float = 2.0):
        """Differentiable weight quantizer. Support different algorithms. Support Quant-Noise with partial quantization.

        Args:
            w_bit (int): quantization bitwidth
            quant_ratio (float, optional): Quantization ratio to support full-precision gradient flow. Defaults to 1.0.
            v_max (float, optional): Maxmimum voltage (exclusive).
        """
        super().__init__()

        self.w_bit = w_bit  # w_bit is the number of quantization level, not bitwidth !

        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logger.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.uniform_q = uniform_quantize(num_levels=w_bit, gradient_clip=True)
        self.v_max = v_max

    def set_quant_ratio(self, quant_ratio=None):
        if quant_ratio is None:
            ### get recommended value
            quant_ratio = [
                None,
                0.2,
                0.3,
                0.4,
                0.5,
                0.55,
                0.6,
                0.7,
                0.8,
                0.83,
                0.86,
                0.89,
                0.92,
                0.95,
                0.98,
                0.99,
                1,
            ][min(self.w_bit, 16)]
        assert 0 <= quant_ratio <= 1, logger.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.quant_ratio = quant_ratio

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(1 - self.quant_ratio)
        else:
            quant_noise_mask = None

        weight = torch.sigmoid(x)  # [0, 1]
        weight_q = self.uniform_q(weight)
        if quant_noise_mask is not None:
            noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
            ### unquantized weights have to follow reparameterization, i.e., tanh
            weight_q = weight + noise

        return weight_q
