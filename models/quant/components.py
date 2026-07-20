from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from models.quant.quantizers import UniformAffineQuantizer
from models.quant.ops import affine_fake_quant_weight, gptq_quantize_linear_weight


@torch.no_grad()
def decompose_svd_branch(
    weight: torch.Tensor,
    *,
    rank: int,
    alpha: float = 1.0,
    bits: int = 4,
    symmetric: bool = True,
    eps: float = 1e-8,
    inputs: Optional[torch.Tensor] = None,
    gptq_block_size: int = 128,
    gptq_damp_percentage: float = 0.01,
    weight_group_size: int = -1,
    num_iterations: int = 0,
    no_gptq: bool = False,
    no_svd_early_stop: bool = False,
) -> Tuple[Optional["LowRankBranch"], torch.Tensor, List[dict]]:
    """SVD decomposition with optional iterative refinement.

    Shared by alpha search (``num_iterations=0``, single SVD for honest ranking)
    and freeze (``num_iterations>0``, iterative refinement like deepcompressor).

    Args:
        no_gptq: If True, use minmax quantization for the residual even when
            calibration inputs are available. Matches DiTAS baseline.
        no_svd_early_stop: If True, run all num_iterations without early-stopping.
            Matches DiTAS's fixed 10-iteration alternating SVD.

    Returns:
        branch: LowRankBranch or None if rank <= 0.
        residual: Quantized residual weight, shape ``[out_feat, in_feat]``.
        iter_trace: List of per-iteration error dicts.
    """
    def _svd_branch(W):
        if rank <= 0:
            return None, torch.zeros_like(W)
        branch = LowRankBranch(W.shape[1], W.shape[0], rank=rank, alpha=alpha, weight=W)
        return branch, branch.get_effective_weight()

    def _quantize(R, inp):
        if inp is not None and not no_gptq:
            return gptq_quantize_linear_weight(
                R, inp, bits=bits, symmetric=symmetric,
                block_size=gptq_block_size, damp_percentage=gptq_damp_percentage,
                eps=eps, group_size=weight_group_size,
            )
        return affine_fake_quant_weight(R, bits=bits, symmetric=symmetric, eps=eps, group_size=weight_group_size)

    branch, L = _svd_branch(weight)
    R = weight - L
    residual = _quantize(R, inputs)
    best_err = ((weight - L - residual) ** 2).mean().item()
    iter_trace = [{
        "iter": 0,
        "svd_error": float(((weight - L) ** 2).mean().cpu()),
        "quant_error": best_err,
    }]

    for k in range(num_iterations):
        T = weight - residual
        cand_branch, cand_L = _svd_branch(T)
        cand_R = weight - cand_L
        cand_residual = _quantize(cand_R, inputs)
        cand_err = ((weight - cand_L - cand_residual) ** 2).mean().item()
        if not no_svd_early_stop and cand_err >= best_err:   # early-stop: no improvement
            break
        branch, residual, best_err = cand_branch, cand_residual, cand_err
        iter_trace.append({
            "iter": k + 1,
            "svd_error": float(((weight - cand_L) ** 2).mean().cpu()),
            "quant_error": cand_err,
        })

    return branch, residual, iter_trace


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
            u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
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
        num_svd_iterations=0,
        weight_group_size=-1,
        group_size=-1,
    ):
        super().__init__()
        self.quantizer = UniformAffineQuantizer(
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
            ch_axis=ch_axis,
            eps=eps,
            group_size=group_size,
        )
        self.rank = rank
        self.alpha = alpha
        self.gptq_block_size = gptq_block_size
        self.gptq_damp_percentage = gptq_damp_percentage
        self.max_gptq_samples = max_gptq_samples
        self.smooth_alpha = smooth_alpha
        self.num_svd_iterations = num_svd_iterations
        self.weight_group_size = weight_group_size
        self.svd_iter_trace: list[dict] = []
        self.branch = None
        self.register_buffer("smooth_scale", None)
        self.register_buffer("act_absmax", None)
        self.register_buffer("residual", None)
        self.register_buffer("_nunchaku_residual", None)  # per-group-per-channel GPTQ for alignment
        self.input_cache = []
        self.raw_input_cache = []       # raw x for nunchaku export GPTQ (before smooth)

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
        self.raw_input_cache = []
        self.svd_iter_trace = []

    @torch.no_grad()
    def build_branch(self, weight, quant_weight=None, smooth_scale=None,
                     no_gptq=False, no_svd_early_stop=False):
        if smooth_scale is not None:
            self.smooth_scale = smooth_scale.detach().to(
                device=weight.device, dtype=weight.dtype)
        elif self.smooth_scale is None:
            self.smooth_scale = torch.ones(
                weight.shape[1], device=weight.device, dtype=weight.dtype)

        smooth_weight = weight * self.smooth_scale.reshape(1, -1)

        inputs = None
        if self.input_cache:
            self.raw_input_cache = list(self.input_cache)   # keep raw x for nunchaku export GPTQ
            inputs = torch.cat(self.input_cache, dim=0).to(
                device=weight.device, dtype=weight.dtype)
            inputs = inputs / self.smooth_scale.reshape(1, -1)

        self.branch, self.residual, self.svd_iter_trace = decompose_svd_branch(
            smooth_weight,
            rank=self.rank,
            alpha=self.alpha,
            bits=self.quantizer.bits,
            symmetric=self.quantizer.symmetric,
            eps=self.quantizer.eps,
            inputs=inputs,
            gptq_block_size=self.gptq_block_size,
            gptq_damp_percentage=self.gptq_damp_percentage,
            weight_group_size=self.weight_group_size,
            num_iterations=self.num_svd_iterations,
            no_gptq=no_gptq,
            no_svd_early_stop=no_svd_early_stop,
        )

        self.input_cache = []
        if self.branch is not None and self.rank > 0:
            self._build_nunchaku_residual(
                weight, self.branch.get_effective_weight(), self.smooth_scale)
        return self.branch

    def _build_nunchaku_residual(self, weight, low_rank, smooth_scale):
        """Compute per-group-per-channel GPTQ residual matching nunchaku export."""
        if self.raw_input_cache is None or len(self.raw_input_cache) == 0:
            self._nunchaku_residual = None
            return

        from models.quant.ops import gptq_per_group_int4

        target = (weight * smooth_scale.reshape(1, -1) - low_rank) / smooth_scale.reshape(1, -1)
        raw_inputs = torch.cat(self.raw_input_cache, dim=0).to(
            device=weight.device, dtype=torch.float32)

        group_size = 64
        q_int, scales = gptq_per_group_int4(
            target.float(), raw_inputs, group_size=group_size, bits=4,
            symmetric=True, block_size=self.gptq_block_size,
            damp_percentage=self.gptq_damp_percentage, eps=self.quantizer.eps,
        )
        out_feat, in_feat = target.shape
        n_groups = in_feat // group_size
        deq = (q_int.float() * scales.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_feat, in_feat))
        self._nunchaku_residual = deq.to(dtype=weight.dtype)

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
            "num_svd_iterations": int(self.num_svd_iterations),
            "svd_iter_actual": len(self.svd_iter_trace),
            "svd_iter_trace": self.svd_iter_trace if self.svd_iter_trace else None,
        }


def build_quant_component(**kwargs):
    return LowRankAffineQuantComponent(**kwargs)
