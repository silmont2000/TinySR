"""Static and dynamic quantizers adapted from ViDiT-Q's base_quantizer.py.

Plain Python dict config (no OmegaConf dependency).
"""

import torch
import torch.nn as nn


class StaticQuantizer(nn.Module):
    """Per-tensor static quantizer for weights.

    Quantization params (delta, zero_point) are calibrated on first forward
    and frozen thereafter.  Supports both symmetric and asymmetric modes.
    """

    def __init__(self, quant_config: dict):
        super().__init__()
        self.n_bits = quant_config['n_bits']
        self.sym = quant_config.get('sym', False)
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)
        self.init_done = False
        if self.sym:
            self.x_absmax = None
        else:
            self.x_max = None
            self.x_min = None

    def forward(self, x: torch.Tensor):
        x_quant = self._quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant

    def _quantize(self, x: torch.Tensor):
        if not self.init_done:
            self._init_quant_params(x)
        x_int = torch.round(x / self.delta) - self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        return x_quant

    @torch.no_grad()
    def _init_quant_params(self, x):
        assert len(x.shape) == 2  # [N_group, -1]
        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            self.x_absmax = (torch.max(self.x_absmax, x_absmax)
                             if self.x_absmax is not None else x_absmax).to(x.device)
            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)
        else:
            x_max = x.max(dim=1)[0]
            x_max[x_max < 0] = 0.
            self.x_max = (torch.max(self.x_max.to(x_max.device), x_max)
                          if self.x_max is not None else x_max)
            x_min = x.min(dim=1)[0]
            x_min[x_min > 0] = 0.
            self.x_min = (torch.min(self.x_min.to(x_min.device), x_min)
                          if self.x_min is not None else x_min)
            delta = (x_max - x_min) / (self.n_levels - 1)
            zero_point = torch.round(x_min / delta) + (self.n_levels / 2)
        delta = torch.clamp(delta, min=1e-6)
        self.delta = delta.unsqueeze(-1)
        self.zero_point = zero_point.unsqueeze(-1)


class DynamicQuantizer(nn.Module):
    """Per-token dynamic quantizer for activations.

    Quantization params (delta, zero_point) are computed online per forward pass.
    Supports both symmetric and asymmetric modes.
    """

    def __init__(self, quant_config: dict):
        super().__init__()
        self.n_bits = quant_config['n_bits']
        self.sym = quant_config.get('sym', False)
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1

    def forward(self, x: torch.Tensor):
        x_quant = self._quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant

    def _quantize(self, x: torch.Tensor):
        assert len(x.shape) == 2
        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)
        else:
            x_max = x.max(dim=1)[0]
            x_max[x_max < 0] = 0.
            x_min = x.min(dim=1)[0]
            x_min[x_min > 0] = 0.
            delta = (x_max - x_min) / (self.n_levels - 1)
            zero_point = torch.round(x_min / delta) + (self.n_levels / 2)
        eps = 1e-8
        delta[delta < eps] = eps
        self.delta = delta.unsqueeze(-1)
        self.zero_point = zero_point.unsqueeze(-1)
        x_int = torch.round(x / self.delta) - self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        return x_quant
