import json
import re

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

EXTRA_SUFFIXES = [
    "proj_out.base_layer",
    "proj_out",
]


def get_target_suffixes(quant_scope):
    if quant_scope == "none":
        return []
    if quant_scope == "ffn_only":
        return FFN_SUFFIXES
    if quant_scope == "attn_only":
        return ATTN_SUFFIXES
    if quant_scope == "dit_full":
        return FFN_SUFFIXES + ATTN_SUFFIXES + EXTRA_SUFFIXES
    raise ValueError(f"Unknown quant_scope: {quant_scope}")


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

        if getattr(self.act_quantizer, "quantizer", None) is not None:
            q_act = self.act_quantizer.quantizer
            if getattr(q_act, "per_channel", False) and getattr(q_act, "ch_axis", None) == -1:
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


def _build_replaced_record(name, weight_quant_kwargs, act_quant_kwargs,
                           weight_quant_kind, act_quant_kind):
    return {
        "name": name,
        "weight_bits": int(weight_quant_kwargs.get("bits", -1)),
        "activation_bits": int(act_quant_kwargs.get("bits", -1)),
        "weight_quant_kind": weight_quant_kind,
        "act_quant_kind": act_quant_kind,
    }


def _replace_one_linear(
    module, name, child, parent, child_name,
    weight_quant_kind, act_quant_kind, weight_quant_kwargs, act_quant_kwargs,
):
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


def replace_linear_with_w4a4(
    module: nn.Module,
    target_suffixes=None,
    skip_keywords=("lora_",),
    weight_quant_kind="affine",
    act_quant_kind="affine",
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
            weight_quant_kind, act_quant_kind,
            weight_quant_kwargs, act_quant_kwargs,
        )
        replaced.append(_build_replaced_record(
            name, weight_quant_kwargs, act_quant_kwargs,
            weight_quant_kind, act_quant_kind,
        ))

    return replaced


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

        _replace_one_linear(
            module, name, child, parent, child_name,
            weight_quant_kind, act_quant_kind,
            weight_quant_kwargs, act_quant_kwargs,
        )
        replaced.append(_build_replaced_record(
            name, weight_quant_kwargs, act_quant_kwargs,
            weight_quant_kind, act_quant_kind,
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
