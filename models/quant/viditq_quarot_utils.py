"""Minimal Hadamard utilities for ViDiT-Q quantization.

Only includes what TinySR needs (had144 for dim=1152/4608).
Adapted from ViDiT-Q's quarot_utils.py.
"""

import math
import os
import torch


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def _get_had144():
    pth_path = os.path.join(os.path.dirname(__file__), "viditq_hadamard", "hadamard_mat.pth")
    d = torch.load(pth_path, weights_only=True)
    return d['had.144.tpal']


def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 144 == 0:
        assert is_pow2(n // 144), f"n={n}, n//144={n//144} is not power of 2"
        K = 144
        hadK = _get_had144().T if transpose else _get_had144()
    elif n % 12 == 0:
        assert is_pow2(n // 12), f"n={n}, n//12={n//12} is not power of 2"
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else:
        assert is_pow2(n), f"n={n} is not divisible by 144 or 12, and not power of 2"
        K = 1
    return hadK, K


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input_t = X.clone().view(-1, n, 1)
    output = input_t.clone()
    while input_t.shape[1] > K:
        input_t = input_t.view(input_t.shape[0], input_t.shape[1] // 2, 2, input_t.shape[2])
        output = output.view(input_t.shape)
        output[:, :, 0, :] = input_t[:, :, 0, :] + input_t[:, :, 1, :]
        output[:, :, 1, :] = input_t[:, :, 0, :] - input_t[:, :, 1, :]
        output = output.view(input_t.shape[0], input_t.shape[1], -1)
        (input_t, output) = (output, input_t)
    del output
    if K > 1:
        input_t = hadK.view(1, K, K).to(input_t) @ input_t
    return input_t.view(X.shape) / torch.tensor(n).sqrt()


def random_hadamard_matrix(size, device):
    Q = torch.randint(low=0, high=2, size=(size,), dtype=torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def get_had12():
    return torch.FloatTensor([
        [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
        [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
        [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
        [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
        [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
        [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
        [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
        [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
        [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
        [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
    ])
