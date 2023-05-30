'''
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-29 23:12:45
'''
"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-05-11 14:43:59
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-29 22:40:31
"""

import math

import torch

__all__ = ["conv_output_size", "dpe_grad_estimator"]


def conv_output_size(in_size, kernel_size, padding=0, stride=1, dilation=1):
    return math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def dpe_grad_estimator(dpe):
    class DPEGradEstimator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, W):
            # W: [bs, k, k] complex, differentiable
            W_hard = dpe(x, differentiable=False)  # [bs, k, k] complex
            return W_hard  # return true value w/o DPE error

        @staticmethod
        def backward(ctx, grad_output):
            # pass the gradient to the soft weight
            grad_input = grad_output.clone()
            return None, grad_input

    return DPEGradEstimator.apply
