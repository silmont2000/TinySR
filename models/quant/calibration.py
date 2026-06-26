"""Calibration orchestration and cache save/load."""
import torch
import torch.nn as nn

from models.quant.components import LowRankAffineQuantComponent, AffineQuantComponent
from models.quant.layers import iter_quant_layers, set_quant_enabled, set_observer_enabled
from models.quant.calibrate import calibrate_all_layers, calibrate_all_layers_cascade


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
):
    """Orchestrate the full calibration pipeline.
    
    Phase 1: FP16 forward over calib data to collect activation statistics.
    Phase 2: Freeze and calibrate (single-pass or layer-cascade).
    
    search_mode: None → use preset smooth_alpha 
                 "grid" → single-pass grid search
                 "cascade" → layer-by-layer cascade freeze
    """
    if quant_scope == "none":
        return

    search = search_mode is not None
    cascade = (search_mode == "cascade")
    compute_error = search

    # Phase 1: FP16 forward to collect activation stats
    set_quant_enabled(transformer, False)
    set_observer_enabled(transformer, True)

    for model_input, timesteps, pooled_prompt_embeds, weight_dtype in calib_data_list:
        cascade_forward_fn(model_input, timesteps, pooled_prompt_embeds, weight_dtype)
        if calib_data_list and calib_data_list[0][0].device.type == "cuda":
            torch.cuda.empty_cache()

    # Phase 2: calibrate each layer
    if cascade:
        cascade_calib = calib_data_list[:min(cascade_calib_images, len(calib_data_list))]
        calibrate_all_layers_cascade(
            transformer, calib_data=cascade_calib, cascade_forward_fn=cascade_forward_fn,
            num_cascade_calib=cascade_calib_images, do_search=True, compute_error=True,
            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size,
        )
    else:
        calibrate_all_layers(
            transformer, do_search=search, compute_error=compute_error,
            smooth_alpha_override=smooth_alpha_override, alpha_grid_size=alpha_grid_size,
        )

    set_quant_enabled(transformer, True)
    set_observer_enabled(transformer, False)
    print(f"[W4A4] calibration finished with {len(calib_data_list)} calibration passes.")


# -------- calibration cache --------

@torch.no_grad()
def save_calib_cache(transformer: nn.Module, path: str):
    """Save per-layer calibration state to disk."""
    cache = {}
    for name, m in iter_quant_layers(transformer):
        entry = {}

        aq = m.act_quantizer
        if isinstance(aq, AffineQuantComponent):
            entry["act_scale"] = aq.quantizer.scale.detach().cpu()
            entry["act_zero_point"] = aq.quantizer.zero_point.detach().cpu()
            entry["act_calibrated"] = aq.quantizer.calibrated

        wq = m.weight_quantizer
        if isinstance(wq, (AffineQuantComponent, LowRankAffineQuantComponent)):
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
        if isinstance(aq, AffineQuantComponent):
            aq.quantizer.scale = entry["act_scale"].to(device=device, dtype=dtype)
            aq.quantizer.zero_point = entry["act_zero_point"].to(device=device, dtype=dtype)
            aq.quantizer.calibrated = entry["act_calibrated"]

        wq = m.weight_quantizer
        if isinstance(wq, (AffineQuantComponent, LowRankAffineQuantComponent)):
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
