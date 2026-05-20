import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.quant.components import LowRankAffineQuantComponent
from models.quant.layers import QuantLinearW4A4, iter_quant_layers


@torch.no_grad()
def pack_int4_weight(
    weight: torch.Tensor,
    symmetric: bool = True,
    group_size: int = None,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    weight = weight.float()
    out, inp = weight.shape
    qmin, qmax = (-8, 7) if symmetric else (0, 15)
    eps = 1e-8

    if group_size is None:
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(eps) / float(qmax)
        zp = None if symmetric else torch.zeros_like(scale)
        q = torch.round(weight / scale).clamp(qmin, qmax).to(torch.int8)
    else:
        padded = inp
        if inp % group_size != 0:
            padded = ((inp + group_size - 1) // group_size) * group_size
            weight = F.pad(weight, (0, padded - inp))
        weight_g = weight.view(out, -1, group_size)
        max_abs = weight_g.abs().amax(dim=2, keepdim=True).clamp_min(eps)
        scale = max_abs / float(qmax)
        q_g = torch.round(weight_g / scale).clamp(qmin, qmax).to(torch.int8)
        q = q_g.view(out, padded)
        zp = None if symmetric else torch.zeros_like(scale)

    q_unsigned = (q + 8).to(torch.uint8)
    if inp % 2 != 0:
        q_unsigned = torch.nn.functional.pad(q_unsigned, (0, 1))
    packed = q_unsigned[:, 0::2] | (q_unsigned[:, 1::2] << 4)
    return packed.to(torch.uint8), scale.to(torch.float16), zp


@torch.no_grad()
def unpack_residual(packed: torch.Tensor, scale: torch.Tensor,
                    out_features: int, in_features: int,
                    dtype: torch.dtype = torch.float16) -> torch.Tensor:
    out, half_in = packed.shape
    q = torch.stack([(packed >> 0) & 0x0F, (packed >> 4)
                    & 0x0F], dim=-1)
    q = q.reshape(out, half_in * 2)
    q = q.to(torch.int8) - 8
    if half_in * 2 > in_features:
        q = q[:, :in_features]
    return q.float().to(dtype) * scale.to(dtype)


@torch.no_grad()
def load_quantized_model_state(transformer: nn.Module, path: str, device=None):
    data = torch.load(path, map_location=device or "cpu")
    missing, unexpected = transformer.load_state_dict(
        data["state_dict"], strict=False)
    if missing:
        print(
            f"[W4A4] load: missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(
            f"[W4A4] load: unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    print(f"[W4A4] quantized model state loaded <- {path}")
    return data.get("replaced_layers"), data.get("quant_meta", []), data.get("model_args", {})
