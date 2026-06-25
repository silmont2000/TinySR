import copy
import time
from typing import Iterable

import torch
from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.models.linear import SVDQW4A4Linear
from torch import nn

from models.tinysr.tinysd3 import TinySD3Transformer2DModel

_W4A4_PACKER = NunchakuWeightPacker(bits=4)


def _normalize_torch_dtype(torch_dtype: torch.dtype) -> torch.dtype:
    if torch_dtype not in (torch.float16, torch.bfloat16):
        return torch.float16
    return torch_dtype


def svd_split_linear_weight(
    weight: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_fp = weight.detach()
    if rank <= 0:
        residual = weight_fp
        proj_down = weight_fp.new_zeros(0, weight_fp.shape[1])
        proj_up = weight_fp.new_zeros(weight_fp.shape[0], 0)
        return residual, proj_down, proj_up

    u, s, vh = torch.linalg.svd(weight_fp.double(), full_matrices=False)
    eff_rank = min(rank, s.numel())
    proj_down = vh[:eff_rank].to(dtype=weight_fp.dtype)
    proj_up = (u[:, :eff_rank] * s[:eff_rank]).to(dtype=weight_fp.dtype)
    low_rank = proj_up @ proj_down
    residual = weight_fp - low_rank
    return residual.contiguous(), proj_down.contiguous(), proj_up.contiguous()


def quantize_linear_weight_w4a4(
    weight: torch.Tensor,
    group_size: int = 64,
    eps: float = 1e-8,
    torch_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp = weight.detach().float().contiguous()
    out_features, in_features = weight_fp.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")

    weight_groups = weight_fp.view(out_features, in_features // group_size, group_size)
    scale = weight_groups.abs().amax(dim=-1).clamp_min(eps) / 7.0
    qweight = torch.round(weight_groups / scale.unsqueeze(-1)).clamp_(-8, 7).to(torch.int32)
    qweight = qweight.view(out_features, in_features).contiguous()

    packed_qweight = _W4A4_PACKER.pack_weight(qweight)
    scale_dtype = _normalize_torch_dtype(torch_dtype or weight.dtype)
    packed_wscales = _W4A4_PACKER.pack_scale(scale.to(dtype=scale_dtype).contiguous(), group_size=group_size)
    return packed_qweight.contiguous(), packed_wscales.contiguous()


def quantize_linear_weight_w4a4_compat(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Kept for backward compatibility with existing call sites.
    return quantize_linear_weight_w4a4(weight, eps=eps)


def pack_low_rank_weight(
    weight: torch.Tensor,
    *,
    down: bool,
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    return _W4A4_PACKER.pack_lowrank_weight(
        weight.detach().to(dtype=_normalize_torch_dtype(torch_dtype)).contiguous(),
        down=down,
    )


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        if src.device == dst.device and src.dtype == dst.dtype:
            dst.copy_(src)
        else:
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


class NunchakuSVDQLinear(SVDQW4A4Linear):
    """Compatibility wrapper around `nunchaku.models.linear.SVDQW4A4Linear`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=bias,
            precision=precision,
            act_unsigned=act_unsigned,
            torch_dtype=_normalize_torch_dtype(torch_dtype),
            device=device,
        )
        self.debug_name: str | None = None
        self.enable_profile = False
        self.profile = {
            "calls": 0,
            "backend_calls": 0,
            "fallback_calls": 0,
            "cpu_calls": 0,
            "low_rank_calls": 0,
            "backend_ms": 0.0,
            "fallback_ms": 0.0,
            "low_rank_ms": 0.0,
        }
        self.exception_count = 0
        self.first_exception: dict[str, str] | None = None

    @property
    def smooth_scale(self) -> torch.Tensor:
        return self.smooth_factor

    def _invalidate_backend(self):
        # The official nunchaku linear does not cache a Python-side backend object.
        return None

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 32) -> "NunchakuSVDQLinear":
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            torch_dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        residual, proj_down, proj_up = svd_split_linear_weight(linear.weight, rank)
        packed_qweight, packed_wscales = quantize_linear_weight_w4a4(
            residual,
            group_size=mod.group_size,
            torch_dtype=mod.torch_dtype,
        )
        _copy_tensor(mod.qweight, packed_qweight)
        _copy_tensor(mod.wscales, packed_wscales)
        mod.smooth_factor.fill_(1)
        mod.smooth_factor_orig.copy_(mod.smooth_factor)
        if mod.bias is not None and linear.bias is not None:
            _copy_tensor(mod.bias, linear.bias.detach())
        if mod.rank > 0:
            packed_down = pack_low_rank_weight(proj_down, down=True, torch_dtype=mod.torch_dtype)
            packed_up = pack_low_rank_weight(proj_up, down=False, torch_dtype=mod.torch_dtype)
            _copy_tensor(mod.proj_down, packed_down)
            _copy_tensor(mod.proj_up, packed_up)
        return mod

    def reset_profile(self):
        for key in self.profile:
            self.profile[key] = 0 if key.endswith("calls") else 0.0
        self.exception_count = 0
        self.first_exception = None

    def _record_exception(self, exc: Exception):
        self.exception_count += 1
        if self.first_exception is None:
            self.first_exception = {
                "type": type(exc).__name__,
                "message": str(exc),
                "layer": self.debug_name or "<unnamed>",
            }

    def _measure_ms(self, fn, device: torch.device):
        if not self.enable_profile:
            return fn(), None

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize(device)
            return out, float(start.elapsed_time(end))

        start = time.perf_counter()
        out = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return out, elapsed_ms

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        def _run_fast():
            original_shape = x.shape[:-1]
            if x.dim() == 3:
                out = super(NunchakuSVDQLinear, self).forward(x, output=output)
            else:
                flat_x = x.reshape(1, -1, x.shape[-1])
                out = super(NunchakuSVDQLinear, self).forward(flat_x)
                out = out.reshape(*original_shape, self.out_features)
            return out

        if not self.enable_profile:
            if x.device.type != "cuda":
                raise RuntimeError("nunchaku.models.linear.SVDQW4A4Linear only supports CUDA execution.")
            return _run_fast()

        self.profile["calls"] += 1
        if self.rank > 0:
            self.profile["low_rank_calls"] += 1

        if x.device.type != "cuda":
            self.profile["cpu_calls"] += 1
            exc = RuntimeError("nunchaku.models.linear.SVDQW4A4Linear only supports CUDA execution.")
            self._record_exception(exc)
            raise exc

        try:
            out, elapsed_ms = self._measure_ms(_run_fast, x.device)
            self.profile["backend_calls"] += 1
            if elapsed_ms is not None:
                self.profile["backend_ms"] += elapsed_ms
            return out
        except Exception as exc:
            self.profile["fallback_calls"] += 1
            self._record_exception(exc)
            raise


NunchakuSVDQW4A4Linear = NunchakuSVDQLinear
NunchakuW4A4Linear = NunchakuSVDQLinear


def load_nunchaku_svdq_linear_state(
    module: NunchakuSVDQLinear,
    state: dict[str, torch.Tensor],
    name: str,
) -> bool:
    qweight_key = f"{name}.qweight"
    wscales_key = f"{name}.wscales"
    smooth_factor_key = f"{name}.smooth_factor"
    smooth_scale_key = f"{name}.weight_quantizer.smooth_scale"
    bias_key = f"{name}.bias"
    proj_down_key = f"{name}.proj_down"
    proj_up_key = f"{name}.proj_up"
    residual_key = f"{name}.weight_quantizer.residual"
    down_key = f"{name}.weight_quantizer.branch.a.weight"
    up_key = f"{name}.weight_quantizer.branch.b.weight"

    if qweight_key in state and wscales_key in state:
        _copy_tensor(module.qweight, state[qweight_key])
        _copy_tensor(module.wscales, state[wscales_key])
    elif residual_key in state:
        packed_qweight, packed_wscales = quantize_linear_weight_w4a4(
            state[residual_key],
            group_size=module.group_size,
            torch_dtype=module.torch_dtype,
        )
        _copy_tensor(module.qweight, packed_qweight)
        _copy_tensor(module.wscales, packed_wscales)
    else:
        return False

    if module.bias is not None and bias_key in state:
        _copy_tensor(module.bias, state[bias_key])

    smooth = state.get(smooth_factor_key, state.get(smooth_scale_key))
    if smooth is None:
        module.smooth_factor.fill_(1)
    else:
        _copy_tensor(module.smooth_factor, smooth)
    module.smooth_factor_orig.copy_(module.smooth_factor)

    if module.rank > 0:
        if proj_down_key in state:
            _copy_tensor(module.proj_down, state[proj_down_key])
        elif down_key in state:
            packed_down = pack_low_rank_weight(state[down_key], down=True, torch_dtype=module.torch_dtype)
            _copy_tensor(module.proj_down, packed_down)
        else:
            module.proj_down.zero_()

        if proj_up_key in state:
            _copy_tensor(module.proj_up, state[proj_up_key])
        elif up_key in state:
            packed_up = pack_low_rank_weight(state[up_key], down=False, torch_dtype=module.torch_dtype)
            _copy_tensor(module.proj_up, packed_up)
        else:
            module.proj_up.zero_()

    module._invalidate_backend()
    return True


def set_nunchaku_profile_enabled(module: nn.Module, enabled: bool) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, NunchakuSVDQLinear):
            child.enable_profile = enabled
            count += 1
    return count


def reset_nunchaku_profile(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, NunchakuSVDQLinear):
            child.reset_profile()
            count += 1
    return count


def summarize_nunchaku_profile(module: nn.Module) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "layers": 0,
        "called_layers": 0,
        "calls": 0,
        "backend_calls": 0,
        "fallback_calls": 0,
        "cpu_calls": 0,
        "low_rank_calls": 0,
        "backend_ms": 0.0,
        "fallback_ms": 0.0,
        "low_rank_ms": 0.0,
        "exception_count": 0,
    }
    first_exception: dict[str, str] | None = None
    for child in module.modules():
        if not isinstance(child, NunchakuSVDQLinear):
            continue
        summary["layers"] += 1
        if child.profile["calls"] > 0:
            summary["called_layers"] += 1
        for key in (
            "calls",
            "backend_calls",
            "fallback_calls",
            "cpu_calls",
            "low_rank_calls",
            "backend_ms",
            "fallback_ms",
            "low_rank_ms",
        ):
            summary[key] += child.profile[key]
        summary["exception_count"] += child.exception_count
        if first_exception is None and child.first_exception is not None:
            first_exception = child.first_exception
    if first_exception is not None:
        summary["first_exception_type"] = first_exception["type"]
        summary["first_exception_message"] = first_exception["message"]
        summary["first_exception_layer"] = first_exception["layer"]
    return summary


def replace_linear_with_nunchaku_w4a4(
    module: nn.Module,
    rank: int = 32,
    target_suffixes: Iterable[str] | None = None,
    skip_keywords: tuple[str, ...] = ("lora_",),
    exclude_keywords: Iterable[str] | None = None,
) -> list[str]:
    replaced = []
    exclude_keywords = tuple(k for k in (exclude_keywords or ()) if k)

    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if any(keyword in name for keyword in skip_keywords):
            continue
        if target_suffixes is not None and not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        if exclude_keywords and any(keyword in name for keyword in exclude_keywords):
            print(f"[NUNCHAKU] skipped excluded layer: {name}")
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = module.get_submodule(parent_name) if parent_name else module
        quant_linear = NunchakuSVDQLinear.from_linear(child, rank=rank)
        quant_linear.debug_name = name
        setattr(parent, child_name, quant_linear)
        replaced.append(name)
        print(f"[NUNCHAKU] replaced linear layer: {name}")

    print(f"[NUNCHAKU] total replaced linear layers: {len(replaced)}")
    return replaced


class NunchakuTinySD3Transformer2DModelSVDQW4A4(nn.Module):
    """TinySR transformer wrapper backed by nunchaku's official SVDQ linear."""

    def __init__(
        self,
        transformer: TinySD3Transformer2DModel,
        *,
        rank: int = 32,
        quantize_in_place: bool = False,
        merge_lora: bool = True,
        target_suffixes: Iterable[str] | None = None,
        skip_keywords: tuple[str, ...] = ("lora_",),
        exclude_keywords: Iterable[str] | None = None,
    ):
        super().__init__()
        model = transformer if quantize_in_place else copy.deepcopy(transformer)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        self.transformer = model.eval()
        self.config = self.transformer.config
        self.replaced_linears = replace_linear_with_nunchaku_w4a4(
            self.transformer,
            rank=rank,
            target_suffixes=target_suffixes,
            skip_keywords=skip_keywords,
            exclude_keywords=exclude_keywords,
        )
        self.rank = rank

    @classmethod
    def from_transformer(
        cls,
        transformer: TinySD3Transformer2DModel,
        **kwargs,
    ) -> "NunchakuTinySD3Transformer2DModelSVDQW4A4":
        return cls(transformer, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs
    ) -> "NunchakuTinySD3Transformer2DModelSVDQW4A4":
        load_kwargs = {
            "subfolder": kwargs.pop("subfolder", "transformer"),
            "torch_dtype": kwargs.pop("torch_dtype", torch.float16),
            "low_cpu_mem_usage": kwargs.pop("low_cpu_mem_usage", False),
            "ignore_mismatched_sizes": kwargs.pop("ignore_mismatched_sizes", True),
            "cache_dir": kwargs.pop("cache_dir", None),
        }
        transformer = TinySD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        device = kwargs.pop("device", None)
        if device is not None:
            transformer = transformer.to(device=device, dtype=load_kwargs["torch_dtype"])
        return cls(transformer, **kwargs)

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


NunchakuTinySD3Transformer2DModelW4A4 = NunchakuTinySD3Transformer2DModelSVDQW4A4

# Backward-compatible aliases for older call sites.
quantize_linear_weight_w8a8 = quantize_linear_weight_w4a4_compat
NunchakuSVDQW8A8Linear = NunchakuSVDQW4A4Linear
NunchakuW8A8Linear = NunchakuW4A4Linear
replace_linear_with_nunchaku_w8a8 = replace_linear_with_nunchaku_w4a4
NunchakuTinySD3Transformer2DModelSVDQW8A8 = NunchakuTinySD3Transformer2DModelSVDQW4A4
NunchakuTinySD3Transformer2DModelW8A8 = NunchakuTinySD3Transformer2DModelW4A4