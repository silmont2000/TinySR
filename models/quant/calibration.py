import torch
import torch.nn as nn

from models.quant.components import LowRankAffineQuantComponent, AffineQuantComponent
from models.quant.layers import iter_quant_layers, set_quant_enabled, set_observer_enabled
from models.quant.freeze import freeze_quant_params, freeze_quant_params_layer_cascade


@torch.no_grad()
def calibrate_and_freeze(
    transformer: nn.Module,
    calib_data_list: list,
    cascade_forward_fn: callable,
    *,
    quant_scope: str = "none",
    search_smooth_alpha: bool = False,
    layer_cascade_smooth_alpha: bool = False,
    cascade_calib_images: int = 4,
    smooth_alpha_override=None,
    svdq_no_error: bool = False,
    latent_tiled_size: int = 64,
    latent_tiled_overlap: int = 8,
    alpha_grid_size: int = 7,
):
    """Calibrate and freeze quantization parameters for the transformer.

    This is a convenience function that wraps the full calibration protocol:
    observer toggling, forward pass, freeze (single-pass or cascade),
    and post-freeze state reset.

    *calib_data_list*: list of (model_input, timesteps, pooled_prompt_embeds, weight_dtype) tuples
                       used for both observer collection (all entries) and cascade forward (first
                       cascade_calib_images entries).
    *cascade_forward_fn*: callable(model_input, timesteps, pooled_prompt_embeds, weight_dtype)
                          that runs a forward pass through the transformer.
    """
    if quant_scope == "none":
        return

    set_quant_enabled(transformer, False)
    set_observer_enabled(transformer, True)

    for model_input, timesteps, pooled_prompt_embeds, weight_dtype in calib_data_list:
        cascade_forward_fn(model_input, timesteps,
                           pooled_prompt_embeds, weight_dtype)

        if calib_data_list and calib_data_list[0][0].device.type == "cuda":
            torch.cuda.empty_cache()

    compute_error = search_smooth_alpha or not svdq_no_error

    if layer_cascade_smooth_alpha:
        cascade_calib = calib_data_list[:min(
            cascade_calib_images, len(calib_data_list))]
        freeze_quant_params_layer_cascade(
            transformer,
            calib_data=cascade_calib,
            cascade_forward_fn=cascade_forward_fn,
            num_cascade_calib=cascade_calib_images,
            search_smooth_alpha=True,
            compute_error=True,
            smooth_alpha_override=smooth_alpha_override,
            alpha_grid_size=alpha_grid_size,
        )
    else:
        freeze_quant_params(
            transformer,
            search_smooth_alpha=search_smooth_alpha,
            compute_error=compute_error,
            smooth_alpha_override=smooth_alpha_override,
            alpha_grid_size=alpha_grid_size,
        )

    set_quant_enabled(transformer, True)
    set_observer_enabled(transformer, False)

    print(
        f"[W4A4] calibration finished with {len(calib_data_list)} calibration passes.")


@torch.no_grad()
def save_calib_cache(transformer: nn.Module, path: str):
    cache = {}
    for name, m in iter_quant_layers(transformer):
        entry = {}

        act_q = m.act_quantizer
        if isinstance(act_q, AffineQuantComponent):
            entry["act_scale"] = act_q.quantizer.scale.detach().cpu()
            entry["act_zero_point"] = act_q.quantizer.zero_point.detach().cpu()
            entry["act_calibrated"] = act_q.quantizer.calibrated

        w_q = m.weight_quantizer
        if isinstance(w_q, (AffineQuantComponent, LowRankAffineQuantComponent)):
            entry["w_scale"] = w_q.quantizer.scale.detach().cpu()
            entry["w_zero_point"] = w_q.quantizer.zero_point.detach().cpu()
            entry["w_calibrated"] = w_q.quantizer.calibrated

        if isinstance(w_q, LowRankAffineQuantComponent):
            if w_q.smooth_scale is not None:
                entry["smooth_scale"] = w_q.smooth_scale.detach().cpu()
            if w_q.residual is not None:
                entry["residual"] = w_q.residual.detach().cpu()
            if w_q.branch is not None and hasattr(w_q.branch, "a") and w_q.branch.a is not None:
                entry["branch_a_weight"] = w_q.branch.a.weight.detach().cpu()
                if hasattr(w_q.branch.b, "weight"):
                    entry["branch_b_weight"] = w_q.branch.b.weight.detach().cpu()

        cache[name] = entry

    torch.save(cache, path)
    print(f"[W4A4] calibration cache saved ({len(cache)} layers) -> {path}")


@torch.no_grad()
def load_calib_cache(transformer: nn.Module, path: str):
    cache = torch.load(path, map_location="cpu")
    for name, m in iter_quant_layers(transformer):
        if name not in cache:
            raise KeyError(
                f"QuantLinearW4A4 '{name}' not found in calibration cache")
        entry = cache[name]

        device = m.weight.device
        dtype = m.weight.dtype

        act_q = m.act_quantizer
        if isinstance(act_q, AffineQuantComponent):
            act_scale = entry["act_scale"].to(device=device, dtype=dtype)
            act_zero = entry["act_zero_point"].to(device=device, dtype=dtype)

            act_q.quantizer.scale = act_scale
            act_q.quantizer.zero_point = act_zero
            act_q.quantizer.calibrated = entry["act_calibrated"]

        w_q = m.weight_quantizer
        if isinstance(w_q, (AffineQuantComponent, LowRankAffineQuantComponent)):
            w_scale = entry["w_scale"].to(device=device, dtype=dtype)
            w_zero = entry["w_zero_point"].to(device=device, dtype=dtype)

            if w_q.quantizer.per_channel:
                ch_axis = w_q.quantizer.ch_axis % m.weight.dim()
                expected = m.weight.shape[ch_axis]
                if w_scale.numel() != expected:
                    raise ValueError(
                        f"[calib_cache mismatch] layer={name}, "
                        f"expected per-channel scale numel={expected}, got {w_scale.numel()}. "
                        f"Please regenerate calib cache with current model/config."
                    )

            w_q.quantizer.scale = w_scale
            w_q.quantizer.zero_point = w_zero
            w_q.quantizer.calibrated = entry["w_calibrated"]

        if isinstance(w_q, LowRankAffineQuantComponent):
            if "smooth_scale" in entry:
                w_q.smooth_scale = entry["smooth_scale"].to(
                    device, dtype=dtype)
            if "residual" in entry:
                w_q.residual = entry["residual"].to(device, dtype=dtype)
            if "branch_a_weight" in entry and w_q.branch is not None:
                w_q.branch.a.weight.copy_(
                    entry["branch_a_weight"].to(device, dtype=dtype))
                if "branch_b_weight" in entry and hasattr(w_q.branch.b, "weight"):
                    w_q.branch.b.weight.copy_(
                        entry["branch_b_weight"].to(device, dtype=dtype))

        m.weight_quantizer.enabled = True
        m.act_quantizer.enabled = True
        m.weight_quantizer.observer_enabled = False
        m.act_quantizer.observer_enabled = False

    print(f"[W4A4] calibration cache loaded ({len(cache)} layers) <- {path}")
