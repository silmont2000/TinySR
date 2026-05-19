import torch
import torch.nn as nn

from models.quant.quantizers import UniformAffineQuantizer
from models.quant.ops import affine_fake_quant_weight, gptq_quantize_linear_weight


class QuantComponent(nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.observer_enabled = True

    @torch.no_grad()
    def collect_stats(self, x):
        return x

    @torch.no_grad()
    def freeze(self):
        self.observer_enabled = False

    @torch.no_grad()
    def reset(self):
        pass

    def forward(self, x):
        return x

    def meta(self):
        return {
            "type": self.__class__.__name__,
            "enabled": bool(self.enabled),
            "observer_enabled": bool(self.observer_enabled),
        }


class AffineQuantComponent(QuantComponent):
    def __init__(
        self,
        bits=4,
        symmetric=True,
        per_channel=False,
        ch_axis=0,
        eps=1e-8,
    ):
        super().__init__()
        self.quantizer = UniformAffineQuantizer(
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
            ch_axis=ch_axis,
            eps=eps,
        )

    @property
    def enabled(self):
        return self.quantizer.enabled

    @enabled.setter
    def enabled(self, value):
        if "quantizer" in self._modules:
            self.quantizer.enabled = value
        else:
            self.__dict__["enabled"] = value

    @property
    def observer_enabled(self):
        return self.quantizer.observer_enabled

    @observer_enabled.setter
    def observer_enabled(self, value):
        if "quantizer" in self._modules:
            self.quantizer.observer_enabled = value
            self.quantizer.observer.enabled = value
        else:
            self.__dict__["observer_enabled"] = value

    @torch.no_grad()
    def collect_stats(self, x):
        self.quantizer.observer(x)
        return x

    @torch.no_grad()
    def freeze(self):
        self.quantizer.calculate_qparams()
        self.observer_enabled = False

    @torch.no_grad()
    def reset(self):
        device = self.quantizer.scale.device
        self.quantizer.observer.min_val = torch.tensor(
            float("inf"), device=device)
        self.quantizer.observer.max_val = torch.tensor(
            float("-inf"), device=device)
        self.quantizer.scale = torch.tensor(1.0, device=device)
        self.quantizer.zero_point = torch.tensor(0.0, device=device)
        self.quantizer.calibrated = False

    def forward(self, x):
        return self.quantizer(x)

    def meta(self):
        scale = self.quantizer.scale.detach()
        return {
            "type": self.__class__.__name__,
            "enabled": bool(self.enabled),
            "observer_enabled": bool(self.observer_enabled),
            "bits": int(self.quantizer.bits),
            "symmetric": bool(self.quantizer.symmetric),
            "per_channel": bool(self.quantizer.per_channel),
            "ch_axis": int(self.quantizer.ch_axis),
            "qmin": int(self.quantizer.qmin),
            "qmax": int(self.quantizer.qmax),
            "calibrated": bool(self.quantizer.calibrated),
            "scale_shape": list(scale.shape),
            "scale_min": float(scale.min().cpu()),
            "scale_max": float(scale.max().cpu()),
        }


class LowRankBranch(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0, weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        if rank == 0:
            self.a = None
            self.b = None
        elif rank < 0:
            self.a = nn.Linear(in_features, out_features, bias=False)
            self.b = nn.Identity()
        else:
            self.a = nn.Linear(in_features, rank, bias=False)
            self.b = nn.Linear(rank, out_features, bias=False)

        self.reset_parameters(weight)

    @torch.no_grad()
    def reset_parameters(self, weight=None):
        if weight is None:
            if self.rank < 0:
                nn.init.zeros_(self.a.weight)
            elif self.rank > 0:
                nn.init.kaiming_uniform_(self.a.weight)
                nn.init.zeros_(self.b.weight)
            return

        if weight.ndim >= 2:
            assert weight.shape[2:].numel(
            ) == 1, "LowRankBranch only supports linear 2D weights"

        weight = weight.view(weight.shape[0], -1)
        device, dtype = weight.device, weight.dtype
        self.to(device=device, dtype=dtype)

        out_features, in_features = weight.shape
        assert self.in_features == in_features, "Input features size mismatch"
        assert self.out_features == out_features, "Output features size mismatch"

        if self.rank < 0:
            self.a.weight.data.copy_(weight)
        elif self.rank > 0:
            u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
            rank = min(self.rank, s.numel())
            us = u[:, :rank] * s[:rank]
            vh = vh[:rank]
            assert not us.isnan().any(), "NaN in U * S"
            assert not vh.isnan().any(), "NaN in V^T"
            assert not us.isinf().any(), "Inf in U * S"
            assert not vh.isinf().any(), "Inf in V^T"
            self.a.weight.data.zero_()
            self.b.weight.data.zero_()
            self.a.weight.data[:rank].copy_(vh.to(dtype))
            self.b.weight.data[:, :rank].copy_(us.to(dtype))

    def get_effective_weight(self):
        if self.rank == 0:
            return None
        if self.rank < 0:
            return self.a.weight
        return self.b.weight @ self.a.weight

    def forward(self, x):
        if self.a is None:
            return None
        return self.alpha * self.b(self.a(x))


class LowRankAffineQuantComponent(QuantComponent):
    def __init__(
        self,
        bits=4,
        symmetric=True,
        per_channel=True,
        ch_axis=0,
        eps=1e-8,
        rank=32,
        alpha=1.0,
        gptq_block_size=128,
        gptq_damp_percentage=0.01,
        max_gptq_samples=2048,
        smooth_alpha=0.5,
    ):
        super().__init__()
        self.quantizer = UniformAffineQuantizer(
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
            ch_axis=ch_axis,
            eps=eps,
        )
        self.rank = rank
        self.alpha = alpha
        self.gptq_block_size = gptq_block_size
        self.gptq_damp_percentage = gptq_damp_percentage
        self.max_gptq_samples = max_gptq_samples
        self.smooth_alpha = smooth_alpha
        self.branch = None
        self.register_buffer("smooth_scale", None)
        self.register_buffer("act_absmax", None)
        self.register_buffer("residual", None)
        self.input_cache = []

    @property
    def enabled(self):
        return self.quantizer.enabled

    @enabled.setter
    def enabled(self, value):
        if "quantizer" in self._modules:
            self.quantizer.enabled = value
        else:
            self.__dict__["enabled"] = value

    @property
    def observer_enabled(self):
        return self.quantizer.observer_enabled

    @observer_enabled.setter
    def observer_enabled(self, value):
        if "quantizer" in self._modules:
            self.quantizer.observer_enabled = value
            self.quantizer.observer.enabled = value
        else:
            self.__dict__["observer_enabled"] = value

    @torch.no_grad()
    def collect_stats(self, x):
        self.quantizer.observer(x)
        return x

    @torch.no_grad()
    def collect_inputs(self, x):
        x = x.detach()
        act_absmax = x.abs().reshape(-1, x.shape[-1]).amax(dim=0)
        if self.act_absmax is None:
            self.act_absmax = act_absmax.cpu()
        else:
            self.act_absmax = torch.maximum(self.act_absmax, act_absmax.cpu())

        if self.max_gptq_samples == 0:
            return
        x = x.reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return
        remaining = self.max_gptq_samples - \
            sum(t.shape[0] for t in self.input_cache)
        if remaining <= 0:
            return
        self.input_cache.append(x[:remaining].cpu())

    @torch.no_grad()
    def freeze(self):
        self.quantizer.calculate_qparams()
        self.observer_enabled = False

    @torch.no_grad()
    def reset(self):
        device = self.quantizer.scale.device
        self.quantizer.observer.min_val = torch.tensor(
            float("inf"), device=device)
        self.quantizer.observer.max_val = torch.tensor(
            float("-inf"), device=device)
        self.quantizer.scale = torch.tensor(1.0, device=device)
        self.quantizer.zero_point = torch.tensor(0.0, device=device)
        self.quantizer.calibrated = False
        self.branch = None
        self.smooth_scale = None
        self.act_absmax = None
        self.residual = None
        self.input_cache = []

    @torch.no_grad()
    def build_branch(self, weight, quant_weight=None, smooth_scale=None):
        if smooth_scale is not None:
            self.smooth_scale = smooth_scale.detach().to(
                device=weight.device, dtype=weight.dtype)
        elif self.smooth_scale is None:
            self.smooth_scale = torch.ones(
                weight.shape[1], device=weight.device, dtype=weight.dtype)

        smooth_weight = weight * self.smooth_scale.reshape(1, -1)

        if self.rank > 0:
            self.branch = LowRankBranch(
                smooth_weight.shape[1],
                smooth_weight.shape[0],
                rank=self.rank,
                alpha=self.alpha,
                weight=smooth_weight,
            )
            L = self.branch.get_effective_weight()
        else:
            self.branch = None
            L = torch.zeros_like(smooth_weight)

        R = smooth_weight - L

        if self.input_cache:
            inputs = torch.cat(self.input_cache, dim=0).to(
                device=weight.device, dtype=weight.dtype)
            inputs = inputs / self.smooth_scale.reshape(1, -1)
            self.residual = gptq_quantize_linear_weight(
                R,
                inputs,
                bits=self.quantizer.bits,
                symmetric=self.quantizer.symmetric,
                block_size=self.gptq_block_size,
                damp_percentage=self.gptq_damp_percentage,
                eps=self.quantizer.eps,
            )
        else:
            self.residual = affine_fake_quant_weight(
                R,
                bits=self.quantizer.bits,
                symmetric=self.quantizer.symmetric,
                eps=self.quantizer.eps,
            )

        self.input_cache = []
        return self.branch

    def forward(self, x):   # x = 原始权重 weight
        if self.observer_enabled:
            self.quantizer.observer(x)      # 收集 min/max 统计
        if not self.enabled:
            return x                        # 不量化, 返回原始权重 (FP16 阶段)
        if self.residual is not None:
            return self.residual            # 已经量化好了, 直接返回缓存 (推理阶段)
        # 还没 build_branch, 临时用普通 fake quant (校准阶段)
        return self.quantizer(x)

    def branch_forward(self, x):
        if not self.enabled:
            return None
        if self.branch is not None:
            return self.branch(x)
        return None

    def meta(self):
        scale = self.quantizer.scale.detach()
        return {
            "type": self.__class__.__name__,
            "enabled": bool(self.enabled),
            "observer_enabled": bool(self.observer_enabled),
            "bits": int(self.quantizer.bits),
            "symmetric": bool(self.quantizer.symmetric),
            "per_channel": bool(self.quantizer.per_channel),
            "ch_axis": int(self.quantizer.ch_axis),
            "qmin": int(self.quantizer.qmin),
            "qmax": int(self.quantizer.qmax),
            "calibrated": bool(self.quantizer.calibrated),
            "scale_shape": list(scale.shape),
            "scale_min": float(scale.min().cpu()),
            "scale_max": float(scale.max().cpu()),
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "gptq_block_size": int(self.gptq_block_size),
            "gptq_damp_percentage": float(self.gptq_damp_percentage),
            "smooth_alpha": float(self.smooth_alpha),
            "has_smooth_scale": self.smooth_scale is not None,
            "has_branch": self.branch is not None,
            "has_residual": self.residual is not None,
        }


def build_quant_component(kind="affine", **kwargs):
    if kind == "none":
        return QuantComponent()
    if kind == "affine":
        return AffineQuantComponent(**kwargs)
    if kind in ("low_rank_affine", "svdq"):
        return LowRankAffineQuantComponent(**kwargs)
    raise ValueError(f"Unknown quant component kind: {kind}")
