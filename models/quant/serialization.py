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


class TorchAOQuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("residual_q_packed", None)
        self.register_buffer("residual_scale", None)
        self.register_buffer("smooth_scale", None)
        self.register_buffer("bias", None)
        self.register_buffer("branch_a_weight", None)
        self.register_buffer("branch_b_weight", None)
        self.branch_alpha: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = unpack_residual(
            self.residual_q_packed, self.residual_scale,
            self.out_features, self.in_features, x.dtype,
        )
        if self.smooth_scale is not None:
            shape = [1] * (x.dim() - 1) + [-1]
            x = x / self.smooth_scale.reshape(*shape)

        out = F.linear(x, residual, self.bias.to(x.dtype)
                       if self.bias is not None else None)

        if self.branch_a_weight is not None and self.branch_b_weight is not None:
            h = F.linear(x, self.branch_a_weight.to(x.dtype))
            h = F.linear(h, self.branch_b_weight.to(x.dtype))
            out = out + self.branch_alpha * h

        return out

    @classmethod
    def from_export(cls, layer_data: dict,
                    device: torch.device = None,
                    dtype: torch.dtype = torch.float16):
        shape = layer_data["residual_shape"]
        out_f, in_f = shape
        mod = cls(in_f, out_f)
        mod.residual_q_packed = layer_data["residual_q_packed"].to(
            device=device)
        mod.residual_scale = layer_data["residual_scale"].to(
            device=device, dtype=dtype)
        if layer_data.get("smooth_scale") is not None:
            mod.smooth_scale = layer_data["smooth_scale"].to(
                device=device, dtype=dtype)
        if layer_data.get("bias") is not None:
            mod.bias = layer_data["bias"].to(device=device, dtype=dtype)
        if "branch_a_weight" in layer_data:
            mod.branch_a_weight = layer_data["branch_a_weight"].to(
                device=device, dtype=dtype)
            mod.branch_b_weight = layer_data["branch_b_weight"].to(
                device=device, dtype=dtype)
            mod.branch_alpha = layer_data.get("branch_alpha", 1.0)
        return mod


@torch.no_grad()
def export_torchao_model(
    transformer: nn.Module,
    path: str,
    replaced_layers: list,
    quant_meta: list,
    model_args: dict = None,
):
    if hasattr(transformer, 'merge_and_unload'):
        print("[torchao] merging LoRA into quantized weights before export ...")
        transformer = transformer.merge_and_unload()
        for _name, m in iter_quant_layers(transformer):
            wq = m.weight_quantizer
            if isinstance(wq, LowRankAffineQuantComponent):
                wq.build_branch(m.weight, smooth_scale=wq.smooth_scale)

    layers_out = {}
    total_size_bytes = 0
    residual_bytes = 0
    branch_bytes = 0
    scale_bytes = 0

    for name, m in iter_quant_layers(transformer):
        wq = m.weight_quantizer
        if not isinstance(wq, LowRankAffineQuantComponent):
            continue
        if wq.residual is None:
            raise RuntimeError(
                f"Layer '{name}' has no residual – run freeze/calibration before export."
            )

        residual = wq.residual
        smooth_scale = wq.smooth_scale
        if smooth_scale is None:
            smooth_scale = torch.ones(
                m.in_features, device=residual.device, dtype=torch.float16)

        packed, scale, zp = pack_int4_weight(residual, symmetric=True)

        ld: dict[str, object] = {
            "residual_q_packed": packed.cpu(),
            "residual_scale": scale.cpu(),
            "residual_shape": list(residual.shape),
            "smooth_scale": smooth_scale.cpu(),
            "bias": m.bias.detach().cpu() if m.bias is not None else None,
        }
        if zp is not None:
            ld["residual_zero_point"] = zp.cpu()

        res_sz = packed.numel() * packed.element_size()
        scl_sz = scale.numel() * scale.element_size()
        total_size_bytes += res_sz + scl_sz
        residual_bytes += res_sz
        scale_bytes += scl_sz

        if wq.branch is not None and hasattr(wq.branch, 'a') and wq.branch.a is not None:
            a_w = wq.branch.a.weight.detach().cpu()
            ld["branch_a_weight"] = a_w
            br_sz = a_w.numel() * a_w.element_size()
            total_size_bytes += br_sz
            branch_bytes += br_sz
            if hasattr(wq.branch.b, 'weight'):
                b_w = wq.branch.b.weight.detach().cpu()
                ld["branch_b_weight"] = b_w
                br_sz2 = b_w.numel() * b_w.element_size()
                total_size_bytes += br_sz2
                branch_bytes += br_sz2
            ld["branch_alpha"] = wq.branch.alpha

        layers_out[name] = ld

    data: dict = {
        "layers": layers_out,
        "replaced_layers": replaced_layers,
        "quant_meta": quant_meta,
    }
    if model_args:
        data["model_args"] = model_args

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(data, path)

    total_mb = total_size_bytes / (1024 * 1024)
    res_mb = residual_bytes / (1024 * 1024)
    br_mb = branch_bytes / (1024 * 1024)
    scl_mb = scale_bytes / (1024 * 1024)
    print(f"[torchao] exported model -> {path}")
    print(f"[torchao]   layers: {len(layers_out)}")
    print(f"[torchao]   total: {total_mb:.1f} MB  "
          f"(residual_packed: {res_mb:.1f} MB, "
          f"branch: {br_mb:.1f} MB, "
          f"scale/etc: {scl_mb:.1f} MB)")


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


@torch.no_grad()
def load_torchao_model(transformer: nn.Module, path: str, device=None):
    data = torch.load(path, map_location=device or "cpu")
    layers_data = data["layers"]

    replaced: list[str] = []
    for export_name, ld in layers_data.items():
        local_name = export_name
        if local_name.endswith(".base_layer"):
            local_name = local_name[:-len(".base_layer")]

        if "." in local_name:
            parent_name, child_name = local_name.rsplit(".", 1)
        else:
            parent_name, child_name = "", local_name

        parent = transformer.get_submodule(
            parent_name) if parent_name else transformer
        child = getattr(parent, child_name, None)
        if child is None:
            print(f"[torchao] WARNING: module '{local_name}' not found – skip")
            continue

        new_mod = TorchAOQuantLinear.from_export(
            ld, device=child.weight.device)
        setattr(parent, child_name, new_mod)
        replaced.append(export_name)

    print(f"[torchao] loaded model <- {path}  ({len(replaced)} layers)")
    return data.get("replaced_layers"), data.get("quant_meta", []), data.get("model_args", {})
