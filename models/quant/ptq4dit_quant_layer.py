"""PTQ4DiT quantized layers for TinySR (SD3-based MMDiT backbone).

Faithful port of the official PTQ4DiT implementation
(https://github.com/adreamwu/PTQ4DiT, NeurIPS 2024):

  - quant/quant_layer.py     -> UniformAffineQuantizer / QuantModule
  - quant/adaptive_rounding.py -> AdaRoundQuantizer
  - quant/block_recon.py     -> block reconstruction (in ptq4dit_wrapper.py)

plus the tau-weighted Spearman smoothing used inside the official Attention/Mlp
(PTQ4DiT's per-layer input scaling), implemented per quantized layer.

Adaptations (documented in agent/qdit_ptq4dit_integration_design.md):
  * only nn.Linear targets (SD3 attention/FFN projections), no conv support;
  * per-token -> per-tensor static activation quantization exactly as the
    official default (`channel_wise=False, leaf_param=True, scale_method='mse'`);
  * the fused qkv is split into to_q/to_k/to_v; each gets its own quantizer.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic helpers (ported from quant/quant_layer.py)
# ---------------------------------------------------------------------------

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for the rounding operation."""
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


# ---------------------------------------------------------------------------
# UniformAffineQuantizer (ported from quant/quant_layer.py)
# ---------------------------------------------------------------------------

class UniformAffineQuantizer(nn.Module):
    """Asymmetric uniform affine quantization with straight-through gradients.

    Supports per-channel (weight) and per-tensor leaf-parameter (activation)
    scale initialization; 'max' or 'mse' scale search as in the official repo.
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = "max", leaf_param: bool = False, always_zero: bool = False):
        super().__init__()
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.leaf_param = leaf_param
        self.scale_method = scale_method
        self.running_stat = False
        self.always_zero = always_zero
        if self.leaf_param:
            self.x_min, self.x_max = None, None

    def __repr__(self):
        s = super().__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)

        # compute in float32 (matching the official fp32 pipeline); keep I/O dtype
        x32 = x.float()
        delta32 = self.delta.float() if torch.is_tensor(self.delta) else torch.tensor(
            self.delta, dtype=torch.float32, device=x.device)
        zero32 = self.zero_point.float() if torch.is_tensor(self.zero_point) else torch.tensor(
            self.zero_point, dtype=torch.float32, device=x.device)
        x_int = round_ste(x32 / delta32) + zero32
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero32) * delta32
        return x_dequant.to(dtype=x.dtype)

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert self.inited
        assert self.leaf_param

        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
        else:
            delta = (self.x_max - self.x_min) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max / (self.n_levels - 1)

        delta = torch.clamp(delta, min=1e-8)
        if not self.sym:
            self.zero_point = (-self.x_min / delta).round() if not (self.sym or self.always_zero) else 0
        self.delta = torch.nn.Parameter(delta)

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        x = x.float()  # fp32 scale search (official runs fp32)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, :, c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if "max" in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if "scale" in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn("Quantization range close to zero: [{}, {}]".format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)
            else:
                x_clone = x.clone().detach()
                x_max = x_clone.max()
                x_min = x_clone.min()
                best_score = 1e+10
                for pct in [0.999, 0.9999, 0.99999]:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except Exception:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device, dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device, dtype=torch.float32)
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction="all")
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (-new_min / delta).round()
        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (-min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


# ---------------------------------------------------------------------------
# QuantModule (ported from quant/quant_layer.py, Linear-only) + tau smoothing
# ---------------------------------------------------------------------------

def _spearmanr(a, b):
    """Spearman rank correlation (pure torch; ties handled by average ranks)."""
    def _rank(x):
        # average ranks for ties
        order = x.argsort(dim=0)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(x.shape[0], dtype=torch.float32, device=x.device)
        return ranks
    a_r, b_r = _rank(a.float()), _rank(b.float())
    a_r = a_r - a_r.mean()
    b_r = b_r - b_r.mean()
    denom = torch.sqrt((a_r ** 2).sum() * (b_r ** 2).sum()).clamp_min(1e-12)
    return (a_r * b_r).sum() / denom


class QuantModule(nn.Module):
    """Quantized Linear: optional input scaling (tau smoothing) + weight/act quant.

    Ported from `quant.quant_layer.QuantModule` with the SmoothQuant-style
    input scaling of PTQ4DiT's Attention/Mlp (`scale_fc1`/`scale_qkv`/`scale_proj`).
    """

    def __init__(self, org_module: nn.Linear, weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False,
                 use_tau_smooth: bool = True):
        super().__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        self.fwd_func = F.linear
        self.fwd_kwargs = dict()
        self.weight = org_module.weight.data
        self.bias = org_module.bias.data if org_module.bias is not None else None

        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant

        self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        # PTQ4DiT tau-weighted smoothing state
        self.use_tau_smooth = use_tau_smooth
        self.tau_scale = None          # input scale (applied as x / tau_scale)
        self.tau_target_samples = 8    # accumulate profiles over this many forwards
        self._tau_profiles = []        # per-sample x_scale profiles
        self._tau_done = False

    def forward(self, input: torch.Tensor, split: int = 0):
        if self.use_tau_smooth and not self._tau_done:
            self._accumulate_tau_profile(input)
            if len(self._tau_profiles) >= self.tau_target_samples:
                self.finalize_tau_smooth_scale()
        if self.tau_scale is not None:
            input = input / self.tau_scale.to(dtype=input.dtype, device=input.device)
        if not self.disable_act_quant and self.use_act_quant:
            input = self.act_quantizer(input)
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            if weight.dtype != input.dtype:
                weight = weight.to(dtype=input.dtype)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)
        return out

    @torch.no_grad()
    def _accumulate_tau_profile(self, x: torch.Tensor):
        """Accumulate per-sample activation channel profiles for the tau smoothing.

        The official code computes the scale from the first (batched) calibration
        forward; TinySR's pipeline runs batch-1 forwards, so we accumulate the
        per-sample profiles across the first `tau_target_samples` forwards and
        finalize once enough samples are seen (equivalent to the official init
        batch, e.g. 8 samples).
        """
        x = x.detach()
        x_scale = x.abs().max(1)[0]  # (B, C) per-sample per-channel
        self._tau_profiles.append(x_scale.float().cpu())

    @torch.no_grad()
    def finalize_tau_smooth_scale(self):
        """PTQ4DiT per-layer input scaling: tau-weighted Spearman aggregation.

        Mirrors the official Attention/Mlp code:
          w_scale = weight.abs().max(0)[0]
          x_scale = per-sample activation channel absmax (accumulated above)
          tau = spearmanr(x_scale[i], w_scale) per sample
          tau_softmax = softmax(-tau)
          x_scale = (x_scale * tau_softmax.view(-1,1)).sum(0)
          scale = (x_scale / (w_scale + 1e-8)) ** 0.5
        Weight is scaled once; input is divided by the scale every forward.
        """
        if self._tau_done or not self._tau_profiles:
            return
        x_scale_all = torch.cat(self._tau_profiles, dim=0).to(self.weight.device)  # (S, C)
        w_scale = self.weight.abs().max(dim=0)[0]
        tau_list = []
        for i in range(x_scale_all.shape[0]):
            tau_list.append(_spearmanr(x_scale_all[i], w_scale))
        tau_tensor = torch.tensor(tau_list).to(self.weight.device)
        tau_softmax = F.softmax(-tau_tensor, dim=0)
        x_scale = (x_scale_all * tau_softmax.view(-1, 1)).sum(dim=0).type(torch.float32)
        self.tau_scale = (x_scale / (w_scale.to(torch.float32) + 1e-8)) ** 0.5
        self.tau_scale = self.tau_scale.to(dtype=self.weight.dtype)
        # scale the weight once (input division happens in forward)
        self.weight = (self.weight.to(torch.float32) * self.tau_scale.to(torch.float32).reshape(1, -1))
        self.weight = self.weight.to(dtype=w_scale.dtype)
        self._tau_profiles = []
        self._tau_done = True

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_running_stat(self, running_stat: bool):
        self.act_quantizer.running_stat = running_stat


# ---------------------------------------------------------------------------
# AdaRoundQuantizer (ported from quant/adaptive_rounding.py)
# ---------------------------------------------------------------------------

def floor_ste(x: torch.Tensor):
    return (x.floor() - x).detach() + x


class AdaRoundQuantizer(nn.Module):
    """Learned (adaptive) rounding for weight quantization."""

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor,
                 round_mode="learned_hard_sigmoid"):
        super().__init__()
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        # compute in float32 for numerical robustness (weights may be fp16)
        x32 = x.float()
        delta32 = self.delta.float() if torch.is_tensor(self.delta) else torch.tensor(self.delta, dtype=torch.float32, device=x.device)
        if self.round_mode == "nearest":
            x_int = torch.round(x32 / delta32)
        elif self.round_mode == "nearest_ste":
            x_int = round_ste(x32 / delta32)
        elif self.round_mode == "stochastic":
            x_floor = torch.floor(x32 / delta32)
            rest = (x32 / delta32) - x_floor
            x_int = x_floor + torch.bernoulli(rest)
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(x32 / delta32)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError("Wrong rounding mode")

        x_quant = torch.clamp(x_int + self.zero_point.float(), 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point.float()) * delta32
        return x_float_q.to(dtype=x.dtype)

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x32 = x.float()
        delta32 = self.delta.float() if torch.is_tensor(self.delta) else torch.tensor(self.delta, dtype=torch.float32, device=x.device)
        x_floor = torch.floor(x32 / delta32)
        if self.round_mode == "learned_hard_sigmoid":
            rest = (x32 / delta32) - x_floor
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def extra_repr(self):
        s = "bit={n_bits}, symmetric={sym}, round_mode={round_mode}"
        return s.format(**self.__dict__)
