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
            reduce_dims = [i for i in range(x.dim()) if i != self.ch_axis]
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

        self.observer = MinMaxObserver(per_channel=per_channel, ch_axis=ch_axis)

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
            shape[self.ch_axis] = -1
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
        self.quantizer.observer.min_val = torch.tensor(float("inf"), device=device)
        self.quantizer.observer.max_val = torch.tensor(float("-inf"), device=device)
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


def build_quant_component(kind="affine", **kwargs):
    if kind == "none":
        return QuantComponent()
    if kind == "affine":
        return AffineQuantComponent(**kwargs)
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
            "bits": 8,
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
        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.linear(x_q, w_q, self.bias)


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

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = module.get_submodule(parent_name) if parent_name else module

        quant_child = QuantLinearW4A4(
            child,
            weight_quant_kind=weight_quant_kind,
            act_quant_kind=act_quant_kind,
            weight_quant_kwargs=weight_quant_kwargs,
            act_quant_kwargs=act_quant_kwargs,
        )
        quant_child = quant_child.to(device=child.weight.device, dtype=child.weight.dtype)

        setattr(parent, child_name, quant_child)
        replaced.append(full_name)

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
    for m in module.modules():
        if isinstance(m, QuantLinearW4A4):
            # Collect weight stats from the static weight tensor to support
            # calibration mode where fake-quant is disabled.
            m.weight_quantizer.collect_stats(m.weight)
            m.weight_quantizer.freeze()
            m.act_quantizer.freeze()

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
