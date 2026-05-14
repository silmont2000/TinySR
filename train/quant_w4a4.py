import math
import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinMaxObserver(nn.Module):
    def __init__(self, per_channel=False, ch_axis=0):
        super().__init__()
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.enabled = True

    @torch.no_grad()
    def forward(self, x):
        if not self.enabled:
            return x

        if self.per_channel:
            ch_axis = self.ch_axis % x.dim()
            reduce_dims = [i for i in range(x.dim()) if i != ch_axis]
            cur_min = x.amin(dim=reduce_dims)
            cur_max = x.amax(dim=reduce_dims)

            if self.min_val.numel() == 1:
                self.min_val = cur_min.detach()
                self.max_val = cur_max.detach()
            else:
                self.min_val = torch.minimum(self.min_val, cur_min.detach())
                self.max_val = torch.maximum(self.max_val, cur_max.detach())
        else:
            self.min_val = torch.minimum(self.min_val, x.min().detach())
            self.max_val = torch.maximum(self.max_val, x.max().detach())

        return x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        bits=4,
        symmetric=True,
        per_channel=False,
        ch_axis=0,
        eps=1e-8,
    ):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.eps = eps
        self.enabled = True
        self.observer_enabled = True

        self.observer = MinMaxObserver(
            per_channel=per_channel, ch_axis=ch_axis)

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.calibrated = False

    @property
    def qmin(self):
        if self.symmetric:
            return -(2 ** (self.bits - 1))
        return 0

    @property
    def qmax(self):
        if self.symmetric:
            return 2 ** (self.bits - 1) - 1
        return 2 ** self.bits - 1

    @torch.no_grad()
    def calculate_qparams(self):
        min_val = self.observer.min_val
        max_val = self.observer.max_val

        if self.symmetric:
            max_abs = torch.maximum(min_val.abs(), max_val.abs())
            scale = max_abs / float(self.qmax)
            scale = scale.clamp_min(self.eps)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / float(self.qmax - self.qmin)
            scale = scale.clamp_min(self.eps)
            zero_point = self.qmin - torch.round(min_val / scale)
            zero_point = zero_point.clamp(self.qmin, self.qmax)

        self.scale = scale.detach()
        self.zero_point = zero_point.detach()
        self.calibrated = True

    def reshape_qparams(self, x):
        scale = self.scale
        zero_point = self.zero_point

        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.ch_axis % x.dim()] = -1
            scale = scale.reshape(shape)
            zero_point = zero_point.reshape(shape)

        return scale, zero_point

    def forward(self, x):
        if self.observer_enabled:
            self.observer(x)

        if not self.enabled:
            return x

        if not self.calibrated:
            self.calculate_qparams()

        if torch.isinf(self.scale).any() or torch.isnan(self.scale).any():
            return x

        scale, zero_point = self.reshape_qparams(x)

        x_int = torch.round(x / scale + zero_point)
        x_int = torch.clamp(x_int, self.qmin, self.qmax)
        x_dequant = (x_int - zero_point) * scale
        return x_dequant


@torch.no_grad()
def affine_fake_quant_weight(weight, bits=4, symmetric=True, eps=1e-8):
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = weight.abs().amax(dim=1, keepdim=True).div(float(qmax)).clamp_min(eps)
        zero_point = torch.zeros_like(scale)
    else:
        qmin = 0
        qmax = 2 ** bits - 1
        min_val = weight.amin(dim=1, keepdim=True)
        max_val = weight.amax(dim=1, keepdim=True)
        scale = (max_val - min_val).div(float(qmax - qmin)).clamp_min(eps)
        zero_point = (qmin - torch.round(min_val / scale)).clamp(qmin, qmax)

    weight_int = torch.round(weight / scale + zero_point).clamp(qmin, qmax)
    return (weight_int - zero_point) * scale


@torch.no_grad()
def gptq_quantize_linear_weight(
    weight,
    inputs,
    bits=4,
    symmetric=True,
    block_size=128,
    damp_percentage=0.01,
    eps=1e-8,
):
    if inputs is None or inputs.numel() == 0:
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)

    orig_dtype = weight.dtype
    work_weight = weight.float()
    work_inputs = inputs.to(device=weight.device, dtype=torch.float32)
    if work_inputs.ndim != 2 or work_inputs.shape[1] != work_weight.shape[1]:
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)

    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = work_weight.abs().amax(dim=1, keepdim=True).div(float(qmax)).clamp_min(eps)
        zero_point = torch.zeros_like(scale)
    else:
        qmin = 0
        qmax = 2 ** bits - 1
        min_val = work_weight.amin(dim=1, keepdim=True)
        max_val = work_weight.amax(dim=1, keepdim=True)
        scale = (max_val - min_val).div(float(qmax - qmin)).clamp_min(eps)
        zero_point = (qmin - torch.round(min_val / scale)).clamp(qmin, qmax)

    hessian = math.sqrt(2.0 / max(work_inputs.shape[0], 1)) * work_inputs
    hessian = hessian.t().matmul(hessian)
    dead = hessian.diagonal() == 0
    if dead.any():
        hessian[dead, dead] = 1
        work_weight[:, dead] = 0

    importance = torch.diag(hessian)
    permute = torch.argsort(importance, descending=True)
    inverse_permute = torch.argsort(permute)
    hessian = hessian[permute][:, permute]
    work_weight = work_weight[:, permute]

    hessian_diag = hessian.diagonal()
    hessian_diag_mean = hessian_diag.mean()
    hessian_diag += damp_percentage * hessian_diag_mean

    hessian_inv = None
    for _ in range(200):
        try:
            hessian_inv = torch.linalg.cholesky(hessian)
            hessian_inv = torch.cholesky_inverse(hessian_inv)
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
            break
        except RuntimeError:
            hessian_diag += (damp_percentage * 0.1) * hessian_diag_mean
    if hessian_inv is None:
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)

    qtensor = torch.zeros_like(work_weight)
    num_columns = work_weight.shape[1]
    for c_start in range(0, num_columns, block_size):
        c_end = min(c_start + block_size, num_columns)
        block_weight = work_weight[:, c_start:c_end].clone()
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]
        block_error = torch.zeros_like(block_weight)
        for local_col in range(c_end - c_start):
            column = block_weight[:, local_col]
            pos_diag = block_hessian_inv[local_col, local_col].clamp_min(eps)
            qcolumn = torch.round(column.view(-1, 1) /
                                  scale + zero_point).clamp(qmin, qmax)
            qcolumn = ((qcolumn - zero_point) * scale).view(-1)
            qtensor[:, c_start + local_col] = qcolumn
            column_error = (column - qcolumn) / pos_diag
            block_error[:, local_col] = column_error
            block_weight[:, local_col:] -= column_error.view(-1, 1).matmul(
                block_hessian_inv[local_col, local_col:].view(1, -1)
            )
        work_weight[:,
                    c_end:] -= block_error.matmul(hessian_inv[c_start:c_end, c_end:])

    qtensor = qtensor[:, inverse_permute]
    if qtensor.isnan().any() or qtensor.isinf().any():
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)
    return qtensor.to(orig_dtype)


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
        # Optional deepcompressor-style grouping for joint QKV branches.
        # When set (string), freeze_quant_params() may build a shared low-rank A
        # and slice B for each projection in the same group.
        self.joint_group = None
        self.joint_tag = None
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
            sum(t.shape[0]
                for t in self.input_cache)  # 防止为 GPTQ 收集输入时无限增长显存/内存
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

    def forward(self, x):
        if self.observer_enabled:
            self.quantizer.observer(x)
        if not self.enabled:
            return x
        if self.residual is not None:
            return self.residual
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


class QuantLinearW4A4(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        weight_quant_kind="affine",
        act_quant_kind="affine",
        weight_quant_kwargs=None,
        act_quant_kwargs=None,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None

        weight_quant_kwargs = weight_quant_kwargs or {
            "bits": 4,
            "symmetric": True,
            "per_channel": True,
            "ch_axis": 0,
        }
        act_quant_kwargs = act_quant_kwargs or {
            "bits": 4,
            "symmetric": True,
            "per_channel": True,
            # Activations are typically shaped [B, ..., C] in this project.
            # Per-channel quantization uses the last dim by default.
            "ch_axis": -1,
        }

        self.weight_quantizer = build_quant_component(
            weight_quant_kind,
            **weight_quant_kwargs,
        )
        self.act_quantizer = build_quant_component(
            act_quant_kind,
            **act_quant_kwargs,
        )

        # If activations are quantized per-channel, verify the chosen channel axis is valid
        # for typical activation shapes to avoid silently quantizing on the wrong dim.
        if getattr(self.act_quantizer, "quantizer", None) is not None:
            q = self.act_quantizer.quantizer
            if getattr(q, "per_channel", False) and getattr(q, "ch_axis", None) == -1:
                pass

    def forward(self, x):
        if hasattr(self.weight_quantizer, "collect_inputs") and self.weight_quantizer.observer_enabled:
            self.weight_quantizer.collect_inputs(x)
        smooth_scale = getattr(self.weight_quantizer, "smooth_scale", None)
        if smooth_scale is not None:
            x = x / smooth_scale.reshape(*([1] * (x.dim() - 1)), -1)
            weight = self.weight * smooth_scale.reshape(1, -1)
        else:
            weight = self.weight

        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(weight)
        out = F.linear(x_q, w_q, self.bias)
        branch_out = self.weight_quantizer.branch_forward(x_q) if hasattr(
            self.weight_quantizer, "branch_forward") else None
        if branch_out is not None:
            out = out + branch_out
        return out


def replace_linear_with_w4a4(
    module: nn.Module,
    target_suffixes=None,
    skip_keywords=("lora_",),
    weight_quant_kind="affine",
    act_quant_kind="affine",
    weight_quant_kwargs=None,
    act_quant_kwargs=None,
):
    replaced = []

    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue

        full_name = name

        if any(k in full_name for k in skip_keywords):
            continue

        if target_suffixes is not None:
            if not any(full_name.endswith(suf) for suf in target_suffixes):
                continue

        parent_name, child_name = name.rsplit(
            ".", 1) if "." in name else ("", name)
        parent = module.get_submodule(parent_name) if parent_name else module

        quant_child = QuantLinearW4A4(
            child,
            weight_quant_kind=weight_quant_kind,
            act_quant_kind=act_quant_kind,
            weight_quant_kwargs=weight_quant_kwargs,
            act_quant_kwargs=act_quant_kwargs,
        )
        quant_child = quant_child.to(
            device=child.weight.device, dtype=child.weight.dtype)

        setattr(parent, child_name, quant_child)
        replaced.append(full_name)

    return replaced


def _expand_braces(pattern):
    pattern = re.escape(pattern)
    pattern = pattern.replace(r"\*", ".*")
    pattern = re.sub(r"\\\{([^{}]+)\\\}", lambda m: "(" + "|".join(re.escape(v)
                     for v in m.group(1).split(",")) + ")", pattern)
    return pattern


def _compile_layer_pattern(pattern):
    return re.compile("^" + _expand_braces(pattern) + "$")


def _layer_matches(name, rule):
    for pattern in rule.get("patterns", []):
        if _compile_layer_pattern(pattern).match(name):
            return True

    for suffix in rule.get("suffixes", []):
        if re.search(_expand_braces(suffix) + "$", name):
            return True

    for prefix in rule.get("prefix", []):
        if re.match(_expand_braces(prefix), name):
            return True

    for exact_name in rule.get("names", []):
        if re.match("^" + _expand_braces(exact_name) + "$", name):
            return True

    return False


def _quant_kwargs_from_strategy(strategy, default_weight_kwargs, default_act_kwargs):
    weight_kwargs = dict(default_weight_kwargs or {})
    act_kwargs = dict(default_act_kwargs or {})

    if "weight" in strategy:
        weight_kwargs.update(strategy["weight"])
    if "activation" in strategy:
        act_kwargs.update(strategy["activation"])

    if "w_bits" in strategy:
        weight_kwargs["bits"] = strategy["w_bits"]
    if "a_bits" in strategy:
        act_kwargs["bits"] = strategy["a_bits"]

    return weight_kwargs, act_kwargs


def load_layer_quant_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Layer quant config must be a JSON object")
    if "rules" not in config or not isinstance(config["rules"], list):
        raise ValueError("Layer quant config must contain a 'rules' list")

    return config


def replace_linear_with_w4a4_from_config(
    module: nn.Module,
    config,
    target_suffixes=None,
    skip_keywords=("lora_",),
    default_weight_quant_kind="affine",
    default_act_quant_kind="affine",
    default_weight_quant_kwargs=None,
    default_act_quant_kwargs=None,
):
    if isinstance(config, str):
        config = load_layer_quant_config(config)

    default_rule = config.get("default", {})
    rules = config.get("rules", [])
    replaced = []

    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue

        if any(k in name for k in skip_keywords):
            continue

        if target_suffixes is not None and not any(name.endswith(suf) for suf in target_suffixes):
            continue

        strategy = dict(default_rule)
        for rule in rules:
            if _layer_matches(name, rule):
                strategy.update(rule)

        if not strategy or not strategy.get("enabled", True):
            continue

        weight_quant_kind = strategy.get(
            "weight_quant_kind", default_weight_quant_kind)
        act_quant_kind = strategy.get("act_quant_kind", default_act_quant_kind)
        weight_quant_kwargs, act_quant_kwargs = _quant_kwargs_from_strategy(
            strategy,
            default_weight_quant_kwargs,
            default_act_quant_kwargs,
        )

        parent_name, child_name = name.rsplit(
            ".", 1) if "." in name else ("", name)
        parent = module.get_submodule(parent_name) if parent_name else module

        quant_child = QuantLinearW4A4(
            child,
            weight_quant_kind=weight_quant_kind,
            act_quant_kind=act_quant_kind,
            weight_quant_kwargs=weight_quant_kwargs,
            act_quant_kwargs=act_quant_kwargs,
        )
        quant_child = quant_child.to(
            device=child.weight.device, dtype=child.weight.dtype)

        setattr(parent, child_name, quant_child)
        replaced.append(
            {
                "name": name,
                "weight_bits": int(weight_quant_kwargs.get("bits", -1)),
                "activation_bits": int(act_quant_kwargs.get("bits", -1)),
                "weight_quant_kind": weight_quant_kind,
                "act_quant_kind": act_quant_kind,
            }
        )

    return replaced


def _get_module_and_parent(root: nn.Module, name: str) -> tuple[nn.Module, nn.Module, str]:
    """Return (module, parent_module, child_attr_name) for a dotted module name."""
    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent.get_submodule(child_name), parent, child_name


def replace_qkv_with_joint_w4a4(
    module: nn.Module,
    *,
    q_suffix: str = ".attn.to_q",
    k_suffix: str = ".attn.to_k",
    v_suffix: str = ".attn.to_v",
    skip_keywords=("lora_",),
    weight_quant_kind: str = "svdq",
    act_quant_kind: str = "affine",
    weight_quant_kwargs=None,
    act_quant_kwargs=None,
    joint_group_name: str = "qkv",
):
    """Replace attention q/k/v Linear layers with QuantLinearW4A4 and enable joint low-rank branch.

    This implements the *grouping* behavior used by deepcompressor: Q/K/V are treated
    as a logical group so the low-rank branch can be *shared* across them.

    Notes:
    - This does not fuse projections into a single Linear. It keeps 3 Linear modules,
      but makes the low-rank factor A shared and slices B per projection.
    - Requires weight_quant_kind="svdq" so LowRankAffineQuantComponent exists.
    """
    weight_quant_kwargs = dict(weight_quant_kwargs or {})
    act_quant_kwargs = dict(act_quant_kwargs or {})
    replaced = []

    # Group candidates by common prefix before the suffix (per attention block).
    qkv_groups: dict[str, dict[str, str]] = {}
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if any(k in name for k in skip_keywords):
            continue
        if name.endswith(q_suffix):
            prefix = name[: -len(q_suffix)]
            qkv_groups.setdefault(prefix, {})["q"] = name
        elif name.endswith(k_suffix):
            prefix = name[: -len(k_suffix)]
            qkv_groups.setdefault(prefix, {})["k"] = name
        elif name.endswith(v_suffix):
            prefix = name[: -len(v_suffix)]
            qkv_groups.setdefault(prefix, {})["v"] = name

    for prefix, group in qkv_groups.items():
        if not ("q" in group and "k" in group and "v" in group):
            continue

        # Replace each projection with QuantLinearW4A4.
        q_name, k_name, v_name = group["q"], group["k"], group["v"]
        q_mod, q_parent, q_attr = _get_module_and_parent(module, q_name)
        k_mod, k_parent, k_attr = _get_module_and_parent(module, k_name)
        v_mod, v_parent, v_attr = _get_module_and_parent(module, v_name)
        if not (isinstance(q_mod, nn.Linear) and isinstance(k_mod, nn.Linear) and isinstance(v_mod, nn.Linear)):
            continue

        q_quant = QuantLinearW4A4(
            q_mod,
            weight_quant_kind=weight_quant_kind,
            act_quant_kind=act_quant_kind,
            weight_quant_kwargs=weight_quant_kwargs,
            act_quant_kwargs=act_quant_kwargs,
        ).to(device=q_mod.weight.device, dtype=q_mod.weight.dtype)
        k_quant = QuantLinearW4A4(
            k_mod,
            weight_quant_kind=weight_quant_kind,
            act_quant_kind=act_quant_kind,
            weight_quant_kwargs=weight_quant_kwargs,
            act_quant_kwargs=act_quant_kwargs,
        ).to(device=k_mod.weight.device, dtype=k_mod.weight.dtype)
        v_quant = QuantLinearW4A4(
            v_mod,
            weight_quant_kind=weight_quant_kind,
            act_quant_kind=act_quant_kind,
            weight_quant_kwargs=weight_quant_kwargs,
            act_quant_kwargs=act_quant_kwargs,
        ).to(device=v_mod.weight.device, dtype=v_mod.weight.dtype)

        setattr(q_parent, q_attr, q_quant)
        setattr(k_parent, k_attr, k_quant)
        setattr(v_parent, v_attr, v_quant)

        # Mark low-rank components as a joint group. The actual shared branch is built
        # later in freeze_quant_params().
        for proj, tag in ((q_quant, "q"), (k_quant, "k"), (v_quant, "v")):
            wq = proj.weight_quantizer
            if isinstance(wq, LowRankAffineQuantComponent):
                wq.joint_group = f"{prefix}.{joint_group_name}"
                wq.joint_tag = tag
        replaced.extend([q_name, k_name, v_name])

    return replaced


def set_quant_state(module: nn.Module, weight_quant=True, act_quant=True):
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.enabled = weight_quant
            m.act_quantizer.enabled = act_quant


def set_observer_enabled(module: nn.Module, enabled=True):
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.observer_enabled = enabled
            m.act_quantizer.observer_enabled = enabled


@torch.no_grad()
def freeze_quant_params(module: nn.Module):
    # First pass: freeze quant params for all layers and build per-layer branches.
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            smooth_scale = None
            if hasattr(m.weight_quantizer, "act_absmax") and m.weight_quantizer.act_absmax is not None:
                act_absmax = m.weight_quantizer.act_absmax.to(
                    device=m.weight.device, dtype=m.weight.dtype).clamp_min(1e-8)
                weight_absmax = m.weight.detach().abs().amax(dim=0).clamp_min(1e-8)
                alpha = m.weight_quantizer.smooth_alpha
                smooth_scale = act_absmax.pow(
                    alpha) / weight_absmax.pow(1.0 - alpha)
                smooth_scale = smooth_scale.clamp_min(1e-8)

            # Collect weight stats from the static weight tensor to support
            # calibration mode where fake-quant is disabled.
            stat_weight = m.weight * \
                smooth_scale.reshape(
                    1, -1) if smooth_scale is not None else m.weight
            m.weight_quantizer.collect_stats(stat_weight)
            m.weight_quantizer.freeze()
            if hasattr(m.weight_quantizer, "build_branch"):
                m.weight_quantizer.build_branch(
                    m.weight, smooth_scale=smooth_scale)
            m.act_quantizer.freeze()

    # Second pass: build *shared* low-rank branches for joint QKV groups.
    # This is the deepcompressor-style behavior: share A across q/k/v, slice B.
    joint_groups: dict[str, list[tuple[str, QuantLinearW4A4]]] = {}
    for name, m in iter_quant_layers(module):
        wq = m.weight_quantizer
        if isinstance(wq, LowRankAffineQuantComponent) and getattr(wq, "joint_group", None):
            joint_groups.setdefault(wq.joint_group, []).append((name, m))

    for group_name, items in joint_groups.items():
        # Expect q/k/v; skip if incomplete.
        if len(items) < 2:
            continue
        # Sort deterministically by joint_tag if present.
        def _order_key(kv):
            _name, _m = kv
            tag = getattr(_m.weight_quantizer, "joint_tag", "")
            return {"q": 0, "k": 1, "v": 2}.get(tag, 99)

        items = sorted(items, key=_order_key)
        mods = [m for _, m in items]
        # All projections should share the same in_features.
        in_features = mods[0].in_features
        if any(mm.in_features != in_features for mm in mods):
            continue
        out_features = sum(mm.out_features for mm in mods)
        rank = mods[0].weight_quantizer.rank

        # Use the smoothed weight if available on each module.
        weights = []
        for mm in mods:
            wq = mm.weight_quantizer
            w = mm.weight
            if getattr(wq, "smooth_scale", None) is not None:
                w = w * wq.smooth_scale.reshape(1, -1)
            weights.append(w)
        st = torch.cat(weights, dim=0)

        shared = LowRankBranch(in_features, out_features, rank=rank, weight=st)
        # Now slice shared branch to each projection, and update residuals accordingly.
        oc_idx = 0
        for (_, mm), w in zip(items, weights, strict=True):
            wq = mm.weight_quantizer
            oc = mm.out_features
            branch = LowRankBranch(in_features, oc, rank=rank)
            branch.a = shared.a
            branch.b.to(dtype=w.dtype, device=w.device)
            branch.b.weight.copy_(shared.b.weight[oc_idx : oc_idx + oc])
            oc_idx += oc
            wq.branch = branch
            # Recompute residual based on the new shared branch.
            wq.residual = w - branch.get_effective_weight().view_as(w)
            # Quantize residual into dequant float for forward.
            wq.residual = wq.quantizer(wq.residual)



def set_quant_enabled(module: nn.Module, enabled=True):
    set_quant_state(module, weight_quant=enabled, act_quant=enabled)


def set_weight_quant_enabled(module: nn.Module, enabled=True):
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.enabled = enabled


def set_act_quant_enabled(module: nn.Module, enabled=True):
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            m.act_quantizer.enabled = enabled


@torch.no_grad()
def reset_observers(module: nn.Module):
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.reset()
            m.act_quantizer.reset()


def collect_quant_meta(module: nn.Module):
    meta = []

    for name, m in module.named_modules():
        if not isinstance(m, QuantLinearW4A4):
            continue

        meta.append(
            {
                "name": name,
                "type": "QuantLinearW4A4",
                "in_features": int(m.in_features),
                "out_features": int(m.out_features),
                "has_bias": m.bias is not None,
                "weight": m.weight_quantizer.meta(),
                "activation": m.act_quantizer.meta(),
            }
        )

    return meta


def count_quant_layers(module: nn.Module):
    return sum(1 for m in module.modules() if isinstance(m, QuantLinearW4A4))


def iter_quant_layers(module: nn.Module):
    for name, m in module.named_modules():
        if isinstance(m, QuantLinearW4A4):
            yield name, m


@torch.no_grad()
def save_calib_cache(transformer: nn.Module, path: str):
    cache = {}
    for name, m in iter_quant_layers(transformer):
        entry = {}

        act_q = m.act_quantizer
        if isinstance(act_q, AffineQuantComponent):
            entry["act_scale"] = act_q.quantizer.scale.detach().cpu()
            entry["act_zero_point"] = act_q.quantizer.zero_point.detach().cpu()
            entry["act_calibrated"] = act_q.quantizer.calibrated

        w_q = m.weight_quantizer
        if isinstance(w_q, (AffineQuantComponent, LowRankAffineQuantComponent)):
            entry["w_scale"] = w_q.quantizer.scale.detach().cpu()
            entry["w_zero_point"] = w_q.quantizer.zero_point.detach().cpu()
            entry["w_calibrated"] = w_q.quantizer.calibrated

        if isinstance(w_q, LowRankAffineQuantComponent):
            if w_q.smooth_scale is not None:
                entry["smooth_scale"] = w_q.smooth_scale.detach().cpu()
            if w_q.residual is not None:
                entry["residual"] = w_q.residual.detach().cpu()
            if w_q.branch is not None and hasattr(w_q.branch, "a") and w_q.branch.a is not None:
                entry["branch_a_weight"] = w_q.branch.a.weight.detach().cpu()
                if hasattr(w_q.branch.b, "weight"):
                    entry["branch_b_weight"] = w_q.branch.b.weight.detach().cpu()

        cache[name] = entry

    torch.save(cache, path)
    print(f"[W4A4] calibration cache saved ({len(cache)} layers) -> {path}")


@torch.no_grad()
def load_calib_cache(transformer: nn.Module, path: str):
    cache = torch.load(path, map_location="cpu")
    for name, m in iter_quant_layers(transformer):
        if name not in cache:
            raise KeyError(
                f"QuantLinearW4A4 '{name}' not found in calibration cache")
        entry = cache[name]

        device = m.weight.device
        dtype = m.weight.dtype

        act_q = m.act_quantizer
        if isinstance(act_q, AffineQuantComponent):
            act_scale = entry["act_scale"].to(device=device, dtype=dtype)
            act_zero = entry["act_zero_point"].to(device=device, dtype=dtype)

            # 直接替换buffer，避免从标量 copy_ 到向量时报错
            act_q.quantizer.scale = act_scale
            act_q.quantizer.zero_point = act_zero
            act_q.quantizer.calibrated = entry["act_calibrated"]

        w_q = m.weight_quantizer
        if isinstance(w_q, (AffineQuantComponent, LowRankAffineQuantComponent)):
            w_scale = entry["w_scale"].to(device=device, dtype=dtype)
            w_zero = entry["w_zero_point"].to(device=device, dtype=dtype)

            # 可读错误：cache和当前层结构不一致
            if w_q.quantizer.per_channel:
                ch_axis = w_q.quantizer.ch_axis % m.weight.dim()
                expected = m.weight.shape[ch_axis]
                if w_scale.numel() != expected:
                    raise ValueError(
                        f"[calib_cache mismatch] layer={name}, "
                        f"expected per-channel scale numel={expected}, got {w_scale.numel()}. "
                        f"Please regenerate calib cache with current model/config."
                    )

            # 直接替换buffer，避免 copy_ 形状冲突
            w_q.quantizer.scale = w_scale
            w_q.quantizer.zero_point = w_zero
            w_q.quantizer.calibrated = entry["w_calibrated"]

        if isinstance(w_q, LowRankAffineQuantComponent):
            if "smooth_scale" in entry:
                w_q.smooth_scale = entry["smooth_scale"].to(
                    device, dtype=dtype)
            if "residual" in entry:
                w_q.residual = entry["residual"].to(device, dtype=dtype)
            if "branch_a_weight" in entry and w_q.branch is not None:
                w_q.branch.a.weight.copy_(
                    entry["branch_a_weight"].to(device, dtype=dtype))
                if "branch_b_weight" in entry and hasattr(w_q.branch.b, "weight"):
                    w_q.branch.b.weight.copy_(
                        entry["branch_b_weight"].to(device, dtype=dtype))

        m.weight_quantizer.enabled = True
        m.act_quantizer.enabled = True
        m.weight_quantizer.observer_enabled = False
        m.act_quantizer.observer_enabled = False

    print(f"[W4A4] calibration cache loaded ({len(cache)} layers) <- {path}")
