"""Per-layer W4A4 calibration: alpha resolution, smooth_scale, SVD branch + GPTQ, freeze."""
import torch
import torch.nn as nn
from typing import Union

from models.quant.ops import affine_fake_quant_weight, gptq_quantize_linear_weight, fake_quant_activation
from models.quant.components import LowRankAffineQuantComponent, decompose_svd_branch
from models.quant.layers import QuantLinearW4A4, iter_quant_layers, set_quant_enabled, set_observer_enabled


# -------- smooth_scale utilities --------

def compute_smooth_scale(act_absmax, weight_absmax, alpha, eps=1e-8):
    """s = act_absmax^α / weight_absmax^(1-α), clamped to eps."""
    s = act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
    return s.clamp_min(eps)


# -------- alpha search & error evaluation --------

@torch.no_grad()
def _eval_quant_error(weight, act_absmax, weight_absmax, weight_quantizer, inputs, alpha,
                      act_bits=8, act_symmetric=True, act_scale=None, act_group_size=-1):
    """Evaluate reconstruction error for a given alpha value.
    
    If inputs is provided: fake-quantizes both activation and weight, then compares
    output against the FP16 reference. Otherwise compares weight quantization error.
    """
    smooth_scale = compute_smooth_scale(act_absmax, weight_absmax, alpha)
    smoothed_weight = weight * smooth_scale.reshape(1, -1)
    bits = weight_quantizer.quantizer.bits
    symmetric = weight_quantizer.quantizer.symmetric
    eps_val = weight_quantizer.quantizer.eps

    if inputs is not None:
        inputs_smoothed = inputs / smooth_scale.reshape(1, -1)
        orig_out = inputs @ weight.T
        rank = weight_quantizer.rank
        wgs = getattr(weight_quantizer, "weight_group_size", -1)
        branch, residual_q, _ = decompose_svd_branch(
            smoothed_weight, rank=rank, alpha=weight_quantizer.alpha,
            bits=bits, symmetric=symmetric, eps=eps_val,
            inputs=inputs_smoothed, gptq_block_size=weight_quantizer.gptq_block_size,
            gptq_damp_percentage=weight_quantizer.gptq_damp_percentage,
            weight_group_size=wgs, num_iterations=0,
        )
        q_inputs = fake_quant_activation(inputs_smoothed, bits=act_bits, symmetric=act_symmetric,
                                         eps=eps_val, scale=act_scale, group_size=act_group_size)
        branch_out = branch(inputs_smoothed) if branch is not None else 0.
        q_out = q_inputs @ residual_q.T + branch_out
        return (orig_out - q_out).pow(2).mean()
    else:
        q_weight = affine_fake_quant_weight(smoothed_weight, bits=bits, symmetric=symmetric, eps=eps_val)
        return (smoothed_weight - q_weight).pow(2).mean()


@torch.no_grad()
def search_alpha(weight, act_absmax, weight_quantizer, input_cache=None,
                 alpha_grid=None, num_grids=7, act_bits=8, act_symmetric=True,
                 act_scale=None, act_group_size=-1):
    """Grid-search over α values, returning the one with minimum reconstruction error."""
    if alpha_grid is None:
        num_grids = max(num_grids, 2)
        alpha_grid = [i / (num_grids - 1) for i in range(num_grids)]

    weight_absmax = weight.detach().abs().amax(dim=0).clamp_min(1e-8)
    act_absmax = act_absmax.to(device=weight.device, dtype=weight.dtype).clamp_min(1e-8)

    inputs_cat = None
    if input_cache is not None and len(input_cache) > 0:
        inputs_cat = torch.cat(input_cache, dim=0).to(device=weight.device, dtype=weight.dtype)

    best_alpha = alpha_grid[len(alpha_grid) // 2]
    best_error = torch.tensor(float("inf"), device=weight.device, dtype=weight.dtype)

    for alpha in alpha_grid:
        error = _eval_quant_error(
            weight=weight, act_absmax=act_absmax, weight_absmax=weight_absmax,
            weight_quantizer=weight_quantizer, inputs=inputs_cat, alpha=alpha,
            act_bits=act_bits, act_symmetric=act_symmetric,
            act_scale=act_scale, act_group_size=act_group_size,
        )
        if error < best_error:
            best_error = error
            best_alpha = alpha

    return best_alpha, best_error.item() if best_error.isfinite() else None


# -------- per-layer calibration --------

@torch.no_grad()
def _resolve_alpha(m, layer_name, smooth_alpha_override, do_search, alpha_grid_size, compute_error):
    """Determine the best α for one layer, compute smooth_scale and reconstruction error.
    
    Returns (smooth_scale, alpha, error_str).  smooth_scale may be None for config-only mode.
    """
    wq = m.weight_quantizer
    has_data = hasattr(wq, "act_absmax") and wq.act_absmax is not None

    if not has_data:
        # Config mode: no calibration data — weight-only smoothing via preset alpha
        alpha = max(getattr(wq, "smooth_alpha", 0.5), 0.0)
        if alpha > 0:
            w_absmax = m.weight.detach().abs().amax(dim=0).clamp_min(1e-8)
            smooth_scale = w_absmax.pow(alpha - 1.0)
        else:
            smooth_scale = None
        return smooth_scale, alpha, "err=N/A"

    # --- has calibration data ---
    act_absmax = wq.act_absmax.to(device=m.weight.device, dtype=m.weight.dtype).clamp_min(1e-8)
    w_absmax = m.weight.detach().abs().amax(dim=0).clamp_min(1e-8)

    # Extract act_quantizer params for error evaluation
    aq = getattr(m, "act_quantizer", None)
    if aq and hasattr(aq, "quantizer"):
        act_bits = aq.quantizer.bits
        act_sym = aq.quantizer.symmetric
        act_scale_val = aq.quantizer.scale.detach().clone()
        act_group_size = getattr(aq.quantizer, "group_size", -1)
    else:
        act_bits, act_sym, act_scale_val, act_group_size = 8, True, None, -1

    input_cache = getattr(wq, "input_cache", None)
    alpha = wq.smooth_alpha
    search_err = None
    alpha_source = "default"

    # Priority: override > search > default
    if smooth_alpha_override and layer_name in smooth_alpha_override:
        alpha = smooth_alpha_override[layer_name]
        wq.smooth_alpha = alpha
        alpha_source = "override"
    elif do_search:
        alpha, search_err = search_alpha(
            weight=m.weight, act_absmax=act_absmax, weight_quantizer=wq,
            input_cache=input_cache, act_bits=act_bits, act_symmetric=act_sym,
            act_scale=act_scale_val, num_grids=alpha_grid_size, act_group_size=act_group_size,
        )
        wq.smooth_alpha = alpha
        alpha_source = "search"

    smooth_scale = compute_smooth_scale(act_absmax, w_absmax, alpha)

    # Reconstruction error (for logging)
    error_str = "err=skipped"
    if compute_error and alpha_source != "search":
        inputs_cat = None
        if input_cache is not None and len(input_cache) > 0:
            inputs_cat = torch.cat(input_cache, dim=0).to(device=m.weight.device, dtype=m.weight.dtype)
        if inputs_cat is not None:
            err = _eval_quant_error(
                weight=m.weight, act_absmax=act_absmax, weight_absmax=w_absmax,
                weight_quantizer=wq, inputs=inputs_cat, alpha=alpha,
                act_bits=act_bits, act_symmetric=act_sym,
                act_scale=act_scale_val, act_group_size=act_group_size,
            )
            error_str = f"err={err.item():.6e}"
    elif alpha_source == "search" and search_err is not None:
        error_str = f"err={search_err:.6e}"

    # Log
    inputs_cat = None
    if input_cache is not None and len(input_cache) > 0:
        inputs_cat = torch.cat(input_cache, dim=0).to(device=m.weight.device, dtype=m.weight.dtype)
    inp_max = inputs_cat.abs().max().item() if inputs_cat is not None else 0.0
    inp_std = inputs_cat.std().item() if inputs_cat is not None else 0.0

    if alpha_source == "search":
        print(f"  [alpha_search] {layer_name} -> alpha={alpha:.2f} {error_str} | "
              f"inp_max={inp_max:.4f} inp_std={inp_std:.4f}")
    elif alpha_source == "override":
        print(f"  [alpha_override] {layer_name} -> alpha={alpha:.2f} {error_str}")
    elif compute_error:
        print(f"  [smooth_err] {layer_name} -> alpha={alpha:.2f} {error_str}")

    return smooth_scale, alpha, error_str


@torch.no_grad()
def calibrate_one_layer(m, layer_name, *, do_search=False, compute_error=True,
                        smooth_alpha_override=None, alpha_grid_size=7):
    """Complete calibration of one QuantLinearW4A4 layer.
    
    1. Resolve optimal α → compute smooth_scale
    2. Freeze act_quantizer (calculate scale from observed min/max)
    3. Build SVD low-rank branch + GPTQ-quantize residual
    4. Freeze weight_quantizer
    """
    wq = m.weight_quantizer

    # Step 1: resolve alpha and compute smooth_scale
    smooth_scale, alpha, error_str = _resolve_alpha(
        m, layer_name, smooth_alpha_override, do_search, alpha_grid_size, compute_error)

    # Step 2: freeze weight quantizer & build SVD branch
    if smooth_scale is not None:
        stat_weight = m.weight * smooth_scale.reshape(1, -1)
        wq.collect_stats(stat_weight)
    wq.freeze()
    if hasattr(wq, "build_branch"):
        wq.build_branch(m.weight, smooth_scale=smooth_scale)

    # Step 3: freeze act quantizer
    m.act_quantizer.freeze()

    return smooth_scale


# -------- batch calibration (single-pass and cascade) --------

@torch.no_grad()
def calibrate_all_layers(module, do_search=False, compute_error=True,
                         smooth_alpha_override=None, alpha_grid_size=7):
    """Single-pass calibration: iterate all QuantLinearW4A4 layers, calibrate each one."""
    from tqdm import tqdm
    alpha_counts: dict[float, int] = {}

    layers = [(name, m) for name, m in module.named_modules() if isinstance(m, QuantLinearW4A4)]
    for name, m in tqdm(layers, desc="[calibrate]"):
        calibrate_one_layer(m, name, do_search=do_search, compute_error=compute_error,
                            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size)
        if do_search:
            alpha = m.weight_quantizer.smooth_alpha
            alpha_counts[alpha] = alpha_counts.get(alpha, 0) + 1

    if not do_search and not smooth_alpha_override:
        total = sum(1 for _, m in iter_quant_layers(module))
        print(f"[W4A4] calibration done ({total} layers, fixed alpha)")
    if do_search and alpha_counts:
        print("[alpha_search] per-layer results:")
        for al, cnt in sorted(alpha_counts.items()):
            print(f"  alpha={al:.2f} -> {cnt} layers")
        total = sum(alpha_counts.values())
        primary = max(alpha_counts, key=alpha_counts.get)
        print(f"  total: {total} layers, most common: alpha={primary:.2f} ({alpha_counts[primary]} layers)")


@torch.no_grad()
def calibrate_all_layers_cascade(module, calib_data, cascade_forward_fn, num_cascade_calib=4,
                                 do_search=False, compute_error=True,
                                 smooth_alpha_override=None, alpha_grid_size=7):
    """Layer-by-layer cascade calibration.
    
    Each layer is frozen sequentially so that layer N sees pre-quantized activations
    from previously calibrated layers (more realistic than single-pass).
    """
    set_observer_enabled(module, False)
    all_layers = list(iter_quant_layers(module))

    alpha_counts: dict[float, int] = {}
    cascade_calib_count = min(num_cascade_calib, len(calib_data))

    for layer_idx, (layer_name, m) in enumerate(all_layers):
        # Reset this layer's stats to collect post-quantized inputs from previous layers
        m.weight_quantizer.act_absmax = None
        m.weight_quantizer.input_cache = []
        m.weight_quantizer.observer_enabled = True
        if hasattr(m, "act_quantizer") and hasattr(m.act_quantizer, "quantizer"):
            obs = m.act_quantizer.quantizer.observer
            obs.min_val.fill_(float("inf"))
            obs.max_val.fill_(float("-inf"))
            obs.enabled = True
            m.act_quantizer.observer_enabled = True
            m.act_quantizer.quantizer.observer_enabled = True

        # Forward pass with previously calibrated layers already quantized
        for i in range(cascade_calib_count):
            cascade_forward_fn(*calib_data[i])

        # Stop observing
        m.weight_quantizer.observer_enabled = False
        if hasattr(m, "act_quantizer"):
            m.act_quantizer.observer_enabled = False

        calibrate_one_layer(m, layer_name, do_search=do_search, compute_error=compute_error,
                            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size)
        if do_search:
            alpha = m.weight_quantizer.smooth_alpha
            alpha_counts[alpha] = alpha_counts.get(alpha, 0) + 1

        m.weight_quantizer.enabled = True
        m.act_quantizer.enabled = True

    if not do_search and not smooth_alpha_override:
        total = sum(1 for _, m in iter_quant_layers(module))
        print(f"[W4A4] cascade calibration done ({total} layers, fixed alpha)")
    if do_search and alpha_counts:
        print("[alpha_search] per-layer results:")
        for al, cnt in sorted(alpha_counts.items()):
            print(f"  alpha={al:.2f} -> {cnt} layers")
        total = sum(alpha_counts.values())
        primary = max(alpha_counts, key=alpha_counts.get)
        print(f"  total: {total} layers, most common: alpha={primary:.2f} ({alpha_counts[primary]} layers)")
