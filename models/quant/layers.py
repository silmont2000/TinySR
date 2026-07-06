import torch
import torch.nn as nn
import torch.nn.functional as F

from models.quant.components import build_quant_component


FFN_SUFFIXES = [
    "ff.net.0.proj.base_layer",
    "ff.net.2.base_layer",
    "ff.net.0.proj",
    "ff.net.2",
]

ATTN_SUFFIXES = [
    "attn.to_q.base_layer",
    "attn.to_k.base_layer",
    "attn.to_v.base_layer",
    "attn.to_out.0.base_layer",
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
]

# EXTRA_SUFFIXES = [
#     "proj_out.base_layer",
#     "proj_out",
# ]
EXTRA_SUFFIXES = [
]


def get_target_suffixes(quant_scope, ffn_blocks=None):
    if quant_scope == "none":
        return []
    if quant_scope == "ffn_only":
        return FFN_SUFFIXES
    if quant_scope == "attn_only":
        base = list(ATTN_SUFFIXES)
        if ffn_blocks is not None:
            base += get_ffn_block_suffixes(ffn_blocks)
        return base
    if quant_scope == "dit_full":
        return FFN_SUFFIXES + ATTN_SUFFIXES + EXTRA_SUFFIXES
    raise ValueError(f"Unknown quant_scope: {quant_scope}")


def get_ffn_block_suffixes(blocks):
    """Return FFN suffixes for specific transformer block indices.

    Supports two formats:
      (old) [0, 2, 3] → both ff.net.0.proj and ff.net.2
      (new) [(0, "both"), (2, "up"), (3, "down")] → per-layer mode

    Mode: "both" (default), "up" (ff.net.0.proj only), "down" (ff.net.2 only).
    """
    suffixes = []
    if not blocks:
        return suffixes
    # Detect format
    first = blocks[0]
    if isinstance(first, (list, tuple)):
        for b, mode in blocks:
            if mode in ("both", "up"):
                suffixes.append(f"transformer_blocks.{b}.ff.net.0.proj")
            if mode in ("both", "down"):
                suffixes.append(f"transformer_blocks.{b}.ff.net.2")
    else:
        for b in blocks:
            suffixes.append(f"transformer_blocks.{b}.ff.net.0.proj")
            suffixes.append(f"transformer_blocks.{b}.ff.net.2")
    return suffixes


def parse_ffn_blocks(arg):
    """Parse comma-separated block spec string. Returns None if empty.

    Formats:
      "1,6,9"              → [(1,"both"), (6,"both"), (9,"both")]
      "1.up,6,9.down"      → [(1,"up"), (6,"both"), (9,"down")]
      "1u,6d"              → [(1,"up"), (6,"down")]   (shorthand)
    Backward-compat: returns list[int] when all elements are bare numbers.
    """
    if arg is None or not str(arg).strip():
        return None
    result = []
    all_bare = True
    for token in str(arg).split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith(".up") or token.endswith("u"):
            mode = "up"
            num = token.rstrip("u")
            if "." in num:
                num = num.split(".")[0]
            all_bare = False
        elif token.endswith(".down") or token.endswith("d"):
            mode = "down"
            num = token.rstrip("d")
            if "." in num:
                num = num.split(".")[0]
            all_bare = False
        else:
            mode = "both"
            num = token
        result.append((int(num), mode))
    # If all bare (backward compat), return list of ints
    if all_bare:
        return [b for b, _ in result]
    return result


class QuantLinearW4A4(nn.Module):
    def __init__(self, linear: nn.Linear,
                 weight_quant_kwargs=None, act_quant_kwargs=None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.detach().clone())
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
            "ch_axis": -1,
        }

        self.weight_quantizer = build_quant_component(**weight_quant_kwargs)
        self.act_quantizer = build_quant_component(**act_quant_kwargs)

        if getattr(self.act_quantizer, "quantizer", None) is not None:
            q_act = self.act_quantizer.quantizer
            if getattr(q_act, "per_channel", False) and getattr(q_act, "ch_axis", None) == -1:
                pass

    def forward(self, x):
        if getattr(self, "_nunchaku_aligned", False) and self.weight_quantizer._nunchaku_residual is not None:
            return self._forward_nunchaku_aligned(x)

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
        branch_out = self.weight_quantizer.branch_forward(x) if hasattr(
            self.weight_quantizer, "branch_forward") else None
        if branch_out is not None:
            out = out + branch_out
        return out

    def _forward_nunchaku_aligned(self, x):
        """Nunchaku-aligned forward: dynamic per-group act quant + per-group residual."""
        wq = self.weight_quantizer
        group_size = getattr(self.act_quantizer.quantizer, "group_size", 64)

        # 1. Dynamic per-group int4 act quantization (matches CUDA kernel)
        n_groups = x.shape[-1] // group_size
        x_view = x.reshape(-1, n_groups, group_size)
        act_scale = x_view.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        x_q = (torch.round(x_view / act_scale) * act_scale).reshape_as(x)

        # 2. Per-group-per-channel GPTQ residual (matches exported qweight + wscales)
        w_q = wq._nunchaku_residual

        out = F.linear(x_q, w_q, self.bias)

        # 3. Lora branch: x @ (A/s)^T @ B^T  (unsmoothed proj_down, matches nunchaku)
        if wq.branch is not None and wq.rank > 0:
            s = wq.smooth_scale
            proj_down = wq.branch.a.weight  # (rank, in)
            proj_down_uns = proj_down / s.reshape(1, -1).clamp_min(1e-8)
            lora_h = F.linear(x, proj_down_uns)        # (N, rank)
            lora_h = lora_h.to(dtype=wq.branch.b.weight.dtype)
            lora_out = F.linear(lora_h, wq.branch.b.weight)  # (N, out)
            out = out + wq.alpha * lora_out

        return out

    def enable_nunchaku_aligned(self):
        self._nunchaku_aligned = True


def _build_replaced_record(name, weight_quant_kwargs, act_quant_kwargs):
    return {
        "name": name,
        "weight_bits": int(weight_quant_kwargs.get("bits", -1)),
        "activation_bits": int(act_quant_kwargs.get("bits", -1)),
    }


def _replace_one_linear(
    module, name, child, parent, child_name,
    weight_quant_kwargs, act_quant_kwargs,
):
    quant_child = QuantLinearW4A4(
        child,
        weight_quant_kwargs=weight_quant_kwargs,
        act_quant_kwargs=act_quant_kwargs,
    )
    quant_child = quant_child.to(
        device=child.weight.device, dtype=child.weight.dtype)
    setattr(parent, child_name, quant_child)


def replace_linear_with_w4a4(
    module: nn.Module,
    target_suffixes=None,
    skip_keywords=("lora_",),
    weight_quant_kwargs=None,
    act_quant_kwargs=None,
):
    weight_quant_kwargs = dict(weight_quant_kwargs or {})
    act_quant_kwargs = dict(act_quant_kwargs or {})
    replaced = []

    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue

        if any(k in name for k in skip_keywords):
            continue

        if target_suffixes is not None:
            if not any(name.endswith(suf) for suf in target_suffixes):
                continue

        parent_name, child_name = name.rsplit(
            ".", 1) if "." in name else ("", name)
        parent = module.get_submodule(parent_name) if parent_name else module

        _replace_one_linear(
            module, name, child, parent, child_name,
            weight_quant_kwargs, act_quant_kwargs,
        )
        replaced.append(_build_replaced_record(
            name, weight_quant_kwargs, act_quant_kwargs,
        ))

    return replaced


def set_quant_state(module: nn.Module, weight_quant=True, act_quant=True):
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.enabled = weight_quant
            m.act_quantizer.enabled = act_quant


def set_quant_enabled(module: nn.Module, enabled=True):
    set_quant_state(module, weight_quant=enabled, act_quant=enabled)


def set_observer_enabled(module: nn.Module, enabled=True):
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.observer_enabled = enabled
            m.act_quantizer.observer_enabled = enabled


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


def load_smooth_alpha_from_report(report_path: str) -> dict[str, float]:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    quant_meta = report.get("quant_meta", [])
    alpha_map: dict[str, float] = {}
    for entry in quant_meta:
        name = entry.get("name")
        weight_meta = entry.get("weight", {})
        if name and "smooth_alpha" in weight_meta:
            alpha_map[name] = float(weight_meta["smooth_alpha"])
    print(
        f"[smooth_alpha] loaded {len(alpha_map)} alphas from report <- {report_path}")
    if alpha_map:
        alphas = sorted(set(alpha_map.values()))
        print(f"[smooth_alpha]  values: {[f'{a:.2f}' for a in alphas]}")
    return alpha_map
