"""ViDiT-Q integration for TinySR: layer replacement + calibration + PTQ."""

import torch
import torch.nn as nn
from models.quant.viditq_viditq_quant_layer import ViDiTQuantizedLinear


def _viditq_quant_config(w_bits, a_bits, w_sym, a_sym):
    """Build quant_config dict matching ViDiT-Q's YAML convention."""
    return {
        'weight': {'n_bits': w_bits, 'sym': w_sym},
        'act':    {'n_bits': a_bits, 'sym': a_sym},
    }


def replace_with_viditq(
    module: nn.Module,
    target_suffixes,
    w_bits=8,
    a_bits=8,
    w_sym=False,      # ViDiT-Q default: asymmetric weight
    a_sym=True,        # ViDiT-Q default: symmetric activation
    alpha=0.5,
):
    """Replace nn.Linear layers with ViDiTQuantizedLinear.

    Only replaces layers whose name ends with any suffix in target_suffixes.
    Skips layers with 'lora_' in the name.
    """
    quant_config = _viditq_quant_config(w_bits, a_bits, w_sym, a_sym)
    replaced = []

    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if 'lora_' in name:
            continue
        if target_suffixes is not None and not any(name.endswith(s) for s in target_suffixes):
            continue

        parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = module.get_submodule(parent_name) if parent_name else module

        q_linear = ViDiTQuantizedLinear(
            in_features=child.in_features,
            out_features=child.out_features,
            bias=child.bias is not None,
            device=child.weight.device,
            quant_config=quant_config,
            fp_module=child,
            alpha=alpha,
        )
        setattr(parent, child_name, q_linear)
        replaced.append(name)

    print(f"[VIDITQ] replaced {len(replaced)} nn.Linear -> ViDiTQuantizedLinear "
          f"(W{w_bits}A{a_bits}, alpha={alpha}, w_sym={w_sym}, a_sym={a_sym})")
    return replaced


@torch.no_grad()
def collect_viditq_act_stats(module, calib_data_list, forward_fn):
    """Collect per-channel activation max for each ViDiTQuantizedLinear layer.

    Temporarily disables quantization (quant_mode=False) so the calibration
    forward uses the original FP weights, not the (as-yet-uncalibrated) quantized path.
    """
    act_max = {}
    hooks = []

    def _make_hook(layer_name):
        def hook(module, input, output):
            x = input[0].detach()
            absmax = x.reshape(-1, x.shape[-1]).abs().max(dim=0)[0]
            if layer_name not in act_max:
                act_max[layer_name] = absmax.cpu()
            else:
                act_max[layer_name] = torch.maximum(act_max[layer_name], absmax.cpu())
        return hook

    # set quant_mode=False on all viditq layers for FP stat collection
    viditq_layers = []
    for name, m in module.named_modules():
        if isinstance(m, ViDiTQuantizedLinear):
            viditq_layers.append(m)
            m.quant_mode = False
            hooks.append(m.register_forward_hook(_make_hook(name)))

    for model_input, ts, ppe, wd in calib_data_list:
        forward_fn(model_input, ts, ppe, wd)

    for h in hooks:
        h.remove()

    # set channel_mask and re-enable quant mode
    for name, m in module.named_modules():
        if isinstance(m, ViDiTQuantizedLinear) and name in act_max:
            m.set_channel_mask(act_max[name].to(device=m.fp_module.weight.device))
            m.quant_mode = True

    print(f"[VIDITQ] collected activation stats for {len(act_max)} layers")
    return act_max


@torch.no_grad()
def finalize_viditq_weights(module):
    """Generate rotation matrices and update quantized weights."""
    count = 0
    for name, m in module.named_modules():
        if not isinstance(m, ViDiTQuantizedLinear):
            continue
        try:
            m.set_rotation_matrix(device=m.fp_module.weight.device)
            m.update_quantized_weight()
        except Exception as e:
            raise RuntimeError(
                f"[VIDITQ] failed to finalize layer '{name}' "
                f"(in={m.in_features}, out={m.out_features}): {e}"
            ) from e
        count += 1
        torch.cuda.empty_cache()
    print(f"[VIDITQ] finalized {count} layers (rotation + weight quant)")
