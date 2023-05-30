import torch
from torch import nn

__all__ = ["SIREN", "Swish"]


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sin()
