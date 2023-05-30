import logging
import math

import numpy as np
import torch
import torch.fft
from pyutils.compute import complex_mult, gen_gaussian_noise
from torch import nn


class BatchTrainableButterfly(nn.Module):
    def __init__(
        self,
        batch_size,
        length,
        wbit=32,
        mode="full",
        shared_phases=None,
        bit_reversal=True,
        enable_last_level_phase_shifter=True,
        coupler_transmission_factor_t=np.sqrt(2) / 2,
        coupler_insertion_loss=0,
        crossing_transmission_factor=1,
        crossing_phase_shift=0,
        phase_noise_std=0,
        device=torch.device("cuda:0"),
    ):
        super().__init__()
        self.length = length
        self.wbit = wbit
        self.n_level = int(np.log2(length))
        self.coupler_transmission_factor_t = coupler_transmission_factor_t
        self.coupler_insertion_loss = coupler_insertion_loss
        self.phase_noise_std = phase_noise_std
        self.mode = mode
        assert mode in {
            "full",
            "full_reverse",
        }, "[E] Only support full and full_reverse"

        self.bit_reversal = bit_reversal
        self.enable_last_level_phase_shifter = enable_last_level_phase_shifter
        self.crossing_transmission_factor = crossing_transmission_factor
        self.crossing_phase_shift = crossing_phase_shift

        self.device = device
        self.phases = (
            nn.Parameter(torch.zeros(*batch_size, self.n_level + 1, length // 2, 2, dtype=torch.float, device=device))
            if shared_phases is None
            else shared_phases.data
        )
        # print(self.phases.shape)
        if "reduce" in mode:
            self.ones = torch.ones(self.n_level, length // 2, dtype=torch.float, device=device)
            self.zeros = torch.zeros(self.n_level, length // 2, dtype=torch.float, device=device)
        self.permutations = ButterflyPermutation(
            length,
            crossing_transmission_factor=crossing_transmission_factor,
            crossing_phase_shift=crossing_phase_shift,
            device=device,
        )
        self.permutation_inverse = {
            "full": False,
            "full_reverse": True,
        }[mode]
        self.eye = torch.eye(self.length, self.length, dtype=torch.cfloat, device=device)

        self.directional_coupler = torch.tensor(
            [[1, 1j], [1j, 1]], device=device
        )  # normalization is handled other place
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.phases, a=0, b=2*np.pi)
        nn.init.uniform_(self.phases, a=-np.pi / 2, b=-np.pi / 2)
        # nn.init.uniform_(self.phases, a=0, b=0)

    def propagate_butterfly(self, phases, x):
        # print(weights.shape)
        # phases:  [*bs, n_level, length // 2, 2] real
        if self.bit_reversal:
            x = self.permutations(x, level=-1)  # [batch, length]
        x = x.view([1] * len(self.phases.shape[:-3]) + list(x.shape))  # [1, ..., 1, batch, length]
        for level in range(self.n_level):
            x = x.reshape(list(x.shape[:-1]) + [self.length // 2, 2])  # [1, batch, length // 2, 2]
            x = x.mul(
                torch.exp(1j * phases[..., level : level + 1, :, :])
            )  # [1, batch, length // 2, 2] * [*bs, 1, length // 2, 2] = [*bs, batch, length//2, 2] complex
            x = x.matmul(self.directional_coupler)  # [*bs, batch, length//2, 2] x [2,2] = [*bs, batch, length//2, 2]
            x = x.flatten(-2)  # [*bs, batch, length]
            if level < self.n_level - 1:
                x = self.permutations(x, level, inverse=self.permutation_inverse)
        if self.enable_last_level_phase_shifter:
            x = x.mul(torch.exp(1j * phases[..., level : level + 1, :, :].flatten(-2)))
        if self.bit_reversal:
            x = self.permutations(x, level=self.n_level - 1)

        return x

    def inject_phase_noise(self, phases, phase_noise_std):
        # noise = phases.data.clone().normal_(0, phase_noise_std).clamp_(-0.15, 0.15)
        phases = phases + gen_gaussian_noise(phases, 0, phase_noise_std)
        return phases

    def set_phase_noise(self, noise_std: float = 0.0) -> None:
        self.phase_noise_std = noise_std

    def build_weight(self):
        return self.forward(self.eye)  # [*batch, length, length] complex

    def forward(self, x):
        shape = x.shape  # [..., length]
        x = x.reshape(-1, self.length)  # [batch, length]
        if self.phase_noise_std > 1e-5:
            phases = self.inject_phase_noise(self.phases, self.phase_noise_std)
        else:
            phases = self.phases
        output = self.propagate_butterfly(phases, x)
        if self.mode in {"full_nonideal", "full_reverse_nonideal"}:
            output = output.reshape(shape)
        else:
            output = output.reshape(self.phases.shape[:-3] + shape) / np.sqrt(self.length)
        return output


class ButterflyPermutation(nn.Module):
    def __init__(self, length, crossing_transmission_factor=1, crossing_phase_shift=0, device=torch.device("cuda:0")):
        super(ButterflyPermutation, self).__init__()
        self.length = length
        self.crossing_transmission_factor = crossing_transmission_factor
        assert 0 <= crossing_transmission_factor <= 1, logging.error(
            f"Transmission factor for waveguide crossings must be within [0, 1], but got {crossing_transmission_factor}"
        )
        self.crossing_phase_shift = crossing_phase_shift
        self.n_level = int(np.log2(self.length)) - 1
        self.device = device

        self.forward_indices, self.backward_indices = self.gen_permutation_indices()
        self.bit_reversal_indices = bitreversal_permutation(self.length)
        self.num_crossings = self.calc_num_crossings(self.forward_indices)
        self.crossings = self.gen_crossings(self.num_crossings)

    def gen_permutation_indices(self):
        # forward indices  [1,2,3,4,5,6,7,8] -> [1,5,2,6,3,7,4,8]
        # barkward indices [1,2,3,4,5,6,7,8] -> [1,3,5,7,2,4,6,8]

        forward_indices, backward_indices = [], []
        initial_indices = torch.arange(0, self.length, dtype=torch.long, device=self.device)

        for level in range(self.n_level):
            block_size = 2 ** (level + 2)
            indices = (
                initial_indices.view(-1, self.length // block_size, 2, block_size // 2)
                .transpose(dim0=-2, dim1=-1)
                .contiguous()
                .view(-1)
            )
            forward_indices.append(indices)

            indices = initial_indices.view(-1, self.length // block_size, block_size)
            indices = torch.cat([indices[..., ::2], indices[..., 1::2]], dim=-1).contiguous().view(-1)
            backward_indices.append(indices)
        return forward_indices, backward_indices

    def calc_num_crossings(self, forward_indices):
        ### num crossings are related to forward indices
        ### for example
        ### from: 0 4 1 5 2 6 3 7
        ### to  : 0 1 2 3 4 5 6 7
        ### get : 0 3 1 2 2 1 3 0
        return [(indices - torch.arange(self.length, device=indices.device)).abs() for indices in forward_indices]

    def gen_crossings(self, num_crossings):
        """
        @description: transfer matrix of cascaded crossings, modeling its insertion loss and phase shift
        @param num_crossings {list of torch.Tensor} number of crossings for all waveguides [length] * n_level
        @return: crossings {list of torch.Tensor} cascaded crossing transfer function [length, 2] * n_level
        """
        ### cascaded crossings (t^n)*(e^(n*phi))
        if self.crossing_phase_shift < 1e-6 and self.crossing_transmission_factor > 1 - 1e-6:
            return None
        crossings = []
        for n_cross in num_crossings:
            n_cross = n_cross.float()
            mag = self.crossing_transmission_factor**n_cross
            phase = n_cross * self.crossing_phase_shift
            crossings.append(torch.stack([mag * phase.cos(), mag * phase.sin()], dim=-1))
        return crossings

    def forward(self, x, level, inverse=False):
        if level == -1 or level == self.n_level:
            output = ButterflyPermutationFunction.apply(x, self.bit_reversal_indices)
        else:
            if inverse == False:
                # output = ButterflyPermutationFunction.apply(x, self.forward_indices[level], self.backward_indices[level])
                output = ButterflyPermutationFunction.apply(x, self.forward_indices[level])
                ## in the original transform, crossings are added after permutation
                if self.crossings is not None:
                    output = complex_mult(self.crossings[level][(None,) * (output.dim() - 2)], output)

            else:
                # output = ButterflyPermutationFunction.apply(x, self.backward_indices[self.n_level-level-1], self.forward_indices[self.n_level-level-1])
                ## in the reversed transform, crossings are added before permutation
                if self.crossings is not None:
                    x = complex_mult(self.crossings[level][(None,) * (x.dim() - 2)], x)
                output = ButterflyPermutationFunction.apply(x, self.backward_indices[self.n_level - level - 1])

        return output


def bitreversal_permutation(n, device=torch.device("cuda:0")):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, "n must be a power of 2"
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return torch.from_numpy(perm.squeeze(0)).to(device)


class ButterflyPermutationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_indices):
        ctx.forward_indices = forward_indices
        if torch.is_complex(input):
            output = input[..., forward_indices]
        else:
            output = input[..., forward_indices, :]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        forward_indices = ctx.forward_indices
        grad_input = grad_output.clone()
        if torch.is_complex(grad_input):
            grad_input[..., forward_indices] = grad_output
        else:
            grad_input[..., forward_indices, :] = grad_output
        return grad_input, None
