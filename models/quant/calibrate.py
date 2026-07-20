"""Per-layer W4A4 calibration: alpha resolution, smooth_scale, SVD branch + GPTQ, freeze.

Also contains the calibration orchestrator (Phase 1 + 2 pipeline) and cache save/load.
"""
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
                      act_bits=8, act_symmetric=True, act_scale=None, act_group_size=-1,
                      gate_weight=None, num_iterations=0, no_gptq=False, no_svd_early_stop=False):
    """Evaluate reconstruction error for a given alpha value.
    
    If inputs is provided: fake-quantizes both activation and weight, then compares
    output against the FP16 reference. Otherwise compares weight quantization error.
    
    gate_weight: optional per-channel weight (e.g. gate_mlp) for channel-weighted MSE.
    num_iterations: GPTQ refinement passes (0 = single-shot SVD). Higher = more accurate
                    error estimate but slower. Default 0 keeps backward compat.
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
            weight_group_size=wgs, num_iterations=num_iterations,
            no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop,
        )
        q_inputs = fake_quant_activation(inputs_smoothed, bits=act_bits, symmetric=act_symmetric,
                                         eps=eps_val, scale=act_scale, group_size=act_group_size)
        branch_out = branch(inputs_smoothed) if branch is not None else 0.
        q_out = q_inputs @ residual_q.T + branch_out
        error = orig_out - q_out
        if gate_weight is not None:
            error = error * gate_weight.to(device=error.device, dtype=error.dtype).view(1, -1)
        return error.pow(2).mean()
    else:
        q_weight = affine_fake_quant_weight(smoothed_weight, bits=bits, symmetric=symmetric, eps=eps_val)
        return (smoothed_weight - q_weight).pow(2).mean()


@torch.no_grad()
def search_alpha(weight, act_absmax, weight_quantizer, input_cache=None,
                 alpha_grid=None, num_grids=7, act_bits=8, act_symmetric=True,
                 act_scale=None, act_group_size=-1, gate_weight=None,
                 num_iterations=0, no_gptq=False, no_svd_early_stop=False):
    """Grid-search over α values, returning the one with minimum reconstruction error.
    
    num_iterations: GPTQ refinement passes during error evaluation (0 = single-shot).
                     Set to e.g. 10 for more accurate but slower search.
    """
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
            gate_weight=gate_weight, num_iterations=num_iterations,
            no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop,
        )
        if error < best_error:
            best_error = error
            best_alpha = alpha

    return best_alpha, best_error.item() if best_error.isfinite() else None


# -------- per-layer calibration --------

@torch.no_grad()
def _resolve_alpha(m, layer_name, smooth_alpha_override, do_search, alpha_grid_size, compute_error,
                   gate_weight=None, no_smooth=False, no_gptq=False, no_svd_early_stop=False):
    """Determine the best α for one layer, compute smooth_scale and reconstruction error.
    
    Returns (smooth_scale, alpha, error_str).  smooth_scale may be None for no_smooth or config-only mode.
    """
    wq = m.weight_quantizer

    if no_smooth:
        return None, 0.0, "no_smooth"
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
        num_svd_iters = getattr(wq, "num_svd_iterations", 0)
        search_iters = max(0, min(num_svd_iters, 10))  # cap for efficiency
        alpha, search_err = search_alpha(
            weight=m.weight, act_absmax=act_absmax, weight_quantizer=wq,
            input_cache=input_cache, act_bits=act_bits, act_symmetric=act_sym,
            act_scale=act_scale_val, num_grids=alpha_grid_size, act_group_size=act_group_size,
            gate_weight=gate_weight, num_iterations=search_iters,
            no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop,
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
                no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop,
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
                        smooth_alpha_override=None, alpha_grid_size=7, gate_weight=None,
                        no_smooth=False, no_gptq=False, no_svd_early_stop=False):
    """Complete calibration of one QuantLinearW4A4 layer.
    
    1. Resolve optimal α → compute smooth_scale
    2. Freeze act_quantizer (calculate scale from observed min/max)
    3. Build SVD low-rank branch + GPTQ-quantize residual
    4. Freeze weight_quantizer
    
    When no_smooth=True, smooth_scale is None (identity) — suitable for pure Min-Max baseline.
    When no_gptq=True, skip Hessian-based GPTQ — use simple minmax instead (DiTAS baseline).
    """
    wq = m.weight_quantizer

    # Step 1: resolve alpha and compute smooth_scale
    smooth_scale, alpha, error_str = _resolve_alpha(
        m, layer_name, smooth_alpha_override, do_search, alpha_grid_size, compute_error,
        gate_weight=gate_weight, no_smooth=no_smooth,
        no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop)

    # Step 2: freeze weight quantizer & build SVD branch
    if smooth_scale is not None:
        stat_weight = m.weight * smooth_scale.reshape(1, -1)
        wq.collect_stats(stat_weight)
    wq.freeze()
    if hasattr(wq, "build_branch"):
        wq.build_branch(m.weight, smooth_scale=smooth_scale,
                        no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop)

    # Step 3: freeze act quantizer
    m.act_quantizer.freeze()

    return smooth_scale


# -------- batch calibration (single-pass and cascade) --------

@torch.no_grad()
def calibrate_all_layers(module, do_search=False, compute_error=True,
                         smooth_alpha_override=None, alpha_grid_size=7,
                         no_smooth=False, no_gptq=False, no_svd_early_stop=False):
    """Single-pass calibration: iterate all QuantLinearW4A4 layers, calibrate each one."""
    from tqdm import tqdm
    alpha_counts: dict[float, int] = {}

    layers = [(name, m) for name, m in module.named_modules() if isinstance(m, QuantLinearW4A4)]
    for name, m in tqdm(layers, desc="[calibrate]"):
        gate_weight = None
        if "ff.net.2" in name:
            block_path = name.rsplit(".ff.net.2", 1)[0]
            parent_block = module.get_submodule(block_path)
            gate_weight = getattr(parent_block, "gate_mlp", None)

        calibrate_one_layer(m, name, do_search=do_search, compute_error=compute_error,
                            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size,
                            gate_weight=gate_weight, no_smooth=no_smooth,
                            no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop)
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
                                 smooth_alpha_override=None, alpha_grid_size=7,
                                 no_smooth=False, no_gptq=False, no_svd_early_stop=False):
    """Layer-by-layer cascade calibration.
    
    Each layer is frozen sequentially so that layer N sees pre-quantized activations
    from previously calibrated layers (more realistic than single-pass).

    Phase 1 act_absmax (from 100 images) is preserved for alpha search accuracy.
    Only input_cache is reset and re-collected via cascade forward to capture
    post-quantized activation distribution for GPTQ error evaluation.

    ff.net.2 layers get gate_mlp-weighted MSE, because AdaLayerNormZero applies
    per-channel output gating (gate_mlp * ff_output) that makes some channels
    much more important than others for the final residual.
    """
    set_observer_enabled(module, False)
    all_layers = list(iter_quant_layers(module))

    alpha_counts: dict[float, int] = {}
    cascade_calib_count = min(num_cascade_calib, len(calib_data))

    for layer_idx, (layer_name, m) in enumerate(all_layers):
        # Reset input_cache to collect post-quantized inputs from previous layers.
        # act_absmax from Phase 1 (100 images) is kept for stable alpha search.
        m.weight_quantizer.input_cache = []
        m.weight_quantizer.observer_enabled = True
        _reset_act_observer(m)

        # Forward pass with previously calibrated layers already quantized
        for i in range(cascade_calib_count):
            cascade_forward_fn(*calib_data[i])

        m.weight_quantizer.observer_enabled = False
        if hasattr(m, "act_quantizer"):
            m.act_quantizer.observer_enabled = False

        # Extract gate_mlp for ff.net.2 (per-channel output gating via AdaLayerNormZero)
        gate_weight = None
        if "ff.net.2" in layer_name:
            block_path = layer_name.rsplit(".ff.net.2", 1)[0]
            parent_block = module.get_submodule(block_path)
            gate_weight = getattr(parent_block, "gate_mlp", None)

        calibrate_one_layer(m, layer_name, do_search=do_search, compute_error=compute_error,
                            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size,
                            gate_weight=gate_weight, no_smooth=no_smooth,
                            no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop)
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


# -------- helpers ────────────────────────────────────────────────────

def _reset_act_observer(m):
    if hasattr(m, "act_quantizer") and hasattr(m.act_quantizer, "quantizer"):
        obs = m.act_quantizer.quantizer.observer
        obs.min_val.fill_(float("inf"))
        obs.max_val.fill_(float("-inf"))
        obs.enabled = True
        m.act_quantizer.observer_enabled = True
        m.act_quantizer.quantizer.observer_enabled = True


# -------- calibration orchestrator (Phase 1 + Phase 2) -------------------

@torch.no_grad()
def run_calibration(
    transformer: nn.Module,
    calib_data_list: list,
    cascade_forward_fn: callable,
    *,
    quant_scope: str = "none",
    search_mode: str = None,
    cascade_calib_images: int = 4,
    smooth_alpha_override=None,
    alpha_grid_size: int = 7,
    no_smooth: bool = False,
    no_gptq: bool = False,
    no_svd_early_stop: bool = False,
):
    """Orchestrate the full calibration pipeline.

    Phase 1: FP16 forward over calib data to collect activation statistics.
    Phase 2: Freeze and calibrate (single-pass or layer-cascade).

    search_mode: None → use preset smooth_alpha
                 "grid" → single-pass grid search
                 "cascade" → layer-by-layer cascade freeze
    no_smooth:    True → skip smooth_scale entirely (identity), Min-Max baseline.
    no_gptq:      True → use minmax instead of GPTQ for residual quantization.
    no_svd_early_stop: True → disable SVD iteration early-stop.
    """
    if quant_scope == "none":
        return

    search = search_mode is not None
    cascade = (search_mode == "cascade")
    compute_error = search

    set_quant_enabled(transformer, False)
    set_observer_enabled(transformer, True)

    for model_input, timesteps, pooled_prompt_embeds, weight_dtype in calib_data_list:
        cascade_forward_fn(model_input, timesteps, pooled_prompt_embeds, weight_dtype)
        if calib_data_list and calib_data_list[0][0].device.type == "cuda":
            torch.cuda.empty_cache()

    if cascade:
        cascade_calib = calib_data_list[:min(cascade_calib_images, len(calib_data_list))]
        calibrate_all_layers_cascade(
            transformer, calib_data=cascade_calib, cascade_forward_fn=cascade_forward_fn,
            num_cascade_calib=cascade_calib_images, do_search=True, compute_error=True,
            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size,
            no_smooth=no_smooth, no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop,
        )
    else:
        calibrate_all_layers(
            transformer, do_search=search, compute_error=compute_error,
            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size,
            no_smooth=no_smooth, no_gptq=no_gptq, no_svd_early_stop=no_svd_early_stop,
        )

    set_quant_enabled(transformer, True)
    set_observer_enabled(transformer, False)
    print(f"[W4A4] calibration finished with {len(calib_data_list)} calibration passes.")


# -------- calibration cache save/load ------------------------------------

@torch.no_grad()
def save_calib_cache(transformer: nn.Module, path: str):
    """Save per-layer calibration state to disk."""
    cache = {}
    for name, m in iter_quant_layers(transformer):
        entry = {}
        aq = m.act_quantizer
        if isinstance(aq, LowRankAffineQuantComponent):
            entry["act_scale"] = aq.quantizer.scale.detach().cpu()
            entry["act_zero_point"] = aq.quantizer.zero_point.detach().cpu()
            entry["act_calibrated"] = aq.quantizer.calibrated

        wq = m.weight_quantizer
        if isinstance(wq, LowRankAffineQuantComponent):
            entry["w_scale"] = wq.quantizer.scale.detach().cpu()
            entry["w_zero_point"] = wq.quantizer.zero_point.detach().cpu()
            entry["w_calibrated"] = wq.quantizer.calibrated

        if isinstance(wq, LowRankAffineQuantComponent):
            if wq.smooth_scale is not None:
                entry["smooth_scale"] = wq.smooth_scale.detach().cpu()
            if wq.residual is not None:
                entry["residual"] = wq.residual.detach().cpu()
            if wq.branch is not None and hasattr(wq.branch, "a") and wq.branch.a is not None:
                entry["branch_a_weight"] = wq.branch.a.weight.detach().cpu()
                if hasattr(wq.branch.b, "weight"):
                    entry["branch_b_weight"] = wq.branch.b.weight.detach().cpu()

        cache[name] = entry

    torch.save(cache, path)
    print(f"[W4A4] calibration cache saved ({len(cache)} layers) -> {path}")


@torch.no_grad()
def load_calib_cache(transformer: nn.Module, path: str):
    """Load per-layer calibration state from disk."""
    cache = torch.load(path, map_location="cpu")
    for name, m in iter_quant_layers(transformer):
        if name not in cache:
            raise KeyError(f"QuantLinearW4A4 '{name}' not found in calibration cache")
        entry = cache[name]

        device = m.weight.device
        dtype = m.weight.dtype

        aq = m.act_quantizer
        if isinstance(aq, LowRankAffineQuantComponent):
            aq.quantizer.scale = entry["act_scale"].to(device=device, dtype=dtype)
            aq.quantizer.zero_point = entry["act_zero_point"].to(device=device, dtype=dtype)
            aq.quantizer.calibrated = entry["act_calibrated"]

        wq = m.weight_quantizer
        if isinstance(wq, LowRankAffineQuantComponent):
            w_scale = entry["w_scale"].to(device=device, dtype=dtype)
            w_zero = entry["w_zero_point"].to(device=device, dtype=dtype)
            if wq.quantizer.per_channel:
                ch_axis = wq.quantizer.ch_axis % m.weight.dim()
                expected = m.weight.shape[ch_axis]
                if w_scale.numel() != expected:
                    raise ValueError(
                        f"[calib_cache mismatch] layer={name}, "
                        f"expected per-channel scale numel={expected}, got {w_scale.numel()}. "
                        f"Please regenerate calib cache with current model/config."
                    )
            wq.quantizer.scale = w_scale
            wq.quantizer.zero_point = w_zero
            wq.quantizer.calibrated = entry["w_calibrated"]

        if isinstance(wq, LowRankAffineQuantComponent):
            if "smooth_scale" in entry:
                wq.smooth_scale = entry["smooth_scale"].to(device, dtype=dtype)
            if "residual" in entry:
                wq.residual = entry["residual"].to(device, dtype=dtype)
            if "branch_a_weight" in entry and wq.branch is not None:
                wq.branch.a.weight.copy_(entry["branch_a_weight"].to(device, dtype=dtype))
                if "branch_b_weight" in entry and hasattr(wq.branch.b, "weight"):
                    wq.branch.b.weight.copy_(entry["branch_b_weight"].to(device, dtype=dtype))

        m.weight_quantizer.enabled = True
        m.act_quantizer.enabled = True
        m.weight_quantizer.observer_enabled = False
        m.act_quantizer.observer_enabled = False

    print(f"[W4A4] calibration cache loaded ({len(cache)} layers) <- {path}")
