import torch
import torch.nn as nn

from typing import Union

from models.quant.ops import affine_fake_quant_weight, gptq_quantize_linear_weight, fake_quant_activation
from models.quant.components import LowRankAffineQuantComponent, LowRankBranch
from models.quant.layers import QuantLinearW4A4, iter_quant_layers, set_quant_enabled, set_observer_enabled


@torch.no_grad()
def _eval_smooth_quant_error(
    weight: torch.Tensor,
    act_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    weight_quantizer: LowRankAffineQuantComponent,
    inputs: Union[torch.Tensor, None],
    alpha: float,
    act_bits: int = 8,
    act_symmetric: bool = True,
    act_scale: Union[torch.Tensor, float, None] = None,
) -> torch.Tensor:
    smooth_scale = act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
    smooth_scale = smooth_scale.clamp_min(1e-8)
    smoothed_weight = weight * smooth_scale.reshape(1, -1)

    bits = weight_quantizer.quantizer.bits
    symmetric = weight_quantizer.quantizer.symmetric
    eps = weight_quantizer.quantizer.eps

    if inputs is not None:
        inputs_smoothed = inputs / smooth_scale.reshape(1, -1)
        orig_out = inputs @ weight.T

        out_features, in_features = weight.shape
        rank = weight_quantizer.rank
        if rank > 0:
            branch = LowRankBranch(
                in_features, out_features,
                rank=rank, alpha=weight_quantizer.alpha,
                weight=smoothed_weight,
            )
            L = branch.get_effective_weight()
        else:
            branch = None
            L = torch.zeros_like(smoothed_weight)
        R = smoothed_weight - L

        gptq_block_size = weight_quantizer.gptq_block_size
        gptq_damp = weight_quantizer.gptq_damp_percentage
        residual_q = gptq_quantize_linear_weight(
            R, inputs_smoothed, bits=bits, symmetric=symmetric,
            block_size=gptq_block_size, damp_percentage=gptq_damp, eps=eps,
        )

        q_inputs = fake_quant_activation(
            inputs_smoothed, bits=act_bits, symmetric=act_symmetric, eps=eps, scale=act_scale)

        branch_out = branch(q_inputs) if branch is not None else 0.
        q_out = q_inputs @ residual_q.T + branch_out
        error = (orig_out - q_out).pow(2).mean()
    else:
        q_weight = affine_fake_quant_weight(
            smoothed_weight, bits=bits, symmetric=symmetric, eps=eps)
        error = (smoothed_weight - q_weight).pow(2).mean()

    return error


@torch.no_grad()
def search_smooth_alpha_for_layer(
    weight: torch.Tensor,
    act_absmax: torch.Tensor,
    weight_quantizer: LowRankAffineQuantComponent,
    input_cache: Union[list[torch.Tensor], None] = None,
    alpha_grid: Union[list[float], None] = None,
    num_grids: int = 7,
    act_bits: int = 8,
    act_symmetric: bool = True,
    act_scale: Union[torch.Tensor, float, None] = None
) -> tuple[float, Union[torch.Tensor, None]]:
    if alpha_grid is None:
        num_grids = max(num_grids, 2)
        alpha_grid = [i / (num_grids - 1) for i in range(num_grids)]

    weight_absmax = weight.detach().abs().amax(dim=0).clamp_min(1e-8)
    act_absmax = act_absmax.to(
        device=weight.device, dtype=weight.dtype).clamp_min(1e-8)

    inputs_cat = None
    if input_cache is not None and len(input_cache) > 0:
        inputs_cat = torch.cat(input_cache, dim=0).to(
            device=weight.device, dtype=weight.dtype)

    best_alpha = alpha_grid[len(alpha_grid) // 2]
    best_error = torch.tensor(
        float("inf"), device=weight.device, dtype=weight.dtype)

    for alpha in alpha_grid:
        error = _eval_smooth_quant_error(
            weight=weight,
            act_absmax=act_absmax,
            weight_absmax=weight_absmax,
            weight_quantizer=weight_quantizer,
            inputs=inputs_cat,
            alpha=alpha,
            act_bits=act_bits,
            act_symmetric=act_symmetric,
            act_scale=act_scale,
        )
        if error < best_error:
            best_error = error
            best_alpha = alpha

    return best_alpha, best_error.item() if best_error.isfinite() else None


@torch.no_grad()
def freeze_one_layer(
    m: QuantLinearW4A4,
    layer_name: str,
    search_smooth_alpha: bool = False,
    compute_error: bool = True,
    smooth_alpha_override=None,
    alpha_stats: dict = None,
    alpha_per_layer: list = None,
    extract_act_qparams: bool = True,
    alpha_grid_size: int = 7,
):
    """Freeze a single QuantLinearW4A4 layer — the shared body across single-pass and cascade.

    Handles: act_absmax extraction, act_quantizer freeze, alpha search/override,
    error computation, smooth_scale build, weight stats collection,
    weight quant freeze, low-rank branch build, act quant freeze.

    When *extract_act_qparams* is True (default), freezes the layer's act_quantizer
    and reads its bits/symmetric/scale. Cascade callers may set it False if they
    pre-freeze the act quantizer.
    """
    smooth_scale = None
    any_override = smooth_alpha_override is not None
    has_act_stats = (
        hasattr(m.weight_quantizer, "act_absmax")
        and m.weight_quantizer.act_absmax is not None
    )

    if has_act_stats:
        act_absmax = m.weight_quantizer.act_absmax.to(
            device=m.weight.device, dtype=m.weight.dtype
        ).clamp_min(1e-8)
        weight_absmax = m.weight.detach().abs().amax(dim=0).clamp_min(1e-8)
        alpha = m.weight_quantizer.smooth_alpha
        alpha_source = "default"

        # Act quantizer: freeze + extract params if requested
        if extract_act_qparams:
            act_quantizer = getattr(m, "act_quantizer", None)
            if act_quantizer and hasattr(act_quantizer, "quantizer") and act_quantizer.observer_enabled:
                act_quantizer.freeze()
                act_bits = act_quantizer.quantizer.bits
                act_sym = act_quantizer.quantizer.symmetric
                act_scale_val = act_quantizer.quantizer.scale.detach().clone()
            else:
                act_bits, act_sym, act_scale_val = 8, True, None
        else:
            act_quantizer = getattr(m, "act_quantizer", None)
            if act_quantizer and hasattr(act_quantizer, "quantizer"):
                act_bits = act_quantizer.quantizer.bits
                act_sym = act_quantizer.quantizer.symmetric
                act_scale_val = act_quantizer.quantizer.scale.detach().clone()
            else:
                act_bits, act_sym, act_scale_val = 8, True, None

        input_cache = getattr(m.weight_quantizer, "input_cache", None)

        if any_override and layer_name in smooth_alpha_override:
            alpha = smooth_alpha_override[layer_name]
            m.weight_quantizer.smooth_alpha = alpha
            alpha_source = "override"
        elif search_smooth_alpha or alpha < 0:
            alpha, search_err = search_smooth_alpha_for_layer(
                weight=m.weight,
                act_absmax=act_absmax,
                weight_quantizer=m.weight_quantizer,
                input_cache=input_cache,
                act_bits=act_bits,
                act_symmetric=act_sym,
                act_scale=act_scale_val,
                num_grids=alpha_grid_size,
            )
            m.weight_quantizer.smooth_alpha = alpha
            alpha_source = "search"
        else:
            search_err = None

        # Assemble calib inputs from cache for error computation
        inputs_cat = None
        if input_cache is not None and len(input_cache) > 0:
            inputs_cat = torch.cat(input_cache, dim=0).to(
                device=m.weight.device, dtype=m.weight.dtype)

        # Compute reconstruction error (skip if search already computed it)
        if compute_error and alpha_source != "search" and inputs_cat is not None:
            err = _eval_smooth_quant_error(
                weight=m.weight,
                act_absmax=act_absmax,
                weight_absmax=weight_absmax,
                weight_quantizer=m.weight_quantizer,
                inputs=inputs_cat,
                alpha=alpha,
                act_bits=act_bits,
                act_symmetric=act_sym,
                act_scale=act_scale_val,
            )
            err_str = f"err={err.item():.6e}"
        elif alpha_source == "search" and search_err is not None:
            err_str = f"err={search_err:.6e}"
        elif not compute_error:
            err_str = "err=skipped"
        else:
            err_str = "err=N/A"

        inp_max = inputs_cat.abs().max().item() if inputs_cat is not None else 0.0
        inp_std = inputs_cat.std().item() if inputs_cat is not None else 0.0

        if alpha_source == "search":
            if alpha_stats is not None:
                alpha_stats[alpha] = alpha_stats.get(alpha, 0) + 1
            if alpha_per_layer is not None:
                alpha_per_layer.append((layer_name, alpha))
            print(
                f"  [alpha_search] {layer_name} -> alpha={alpha:.2f} {err_str} | "
                f"inp_max={inp_max:.4f} inp_std={inp_std:.4f}")
        elif alpha_source == "override":
            print(
                f"  [alpha_override] {layer_name} -> alpha={alpha:.2f} {err_str}")
        elif compute_error:
            print(
                f"  [smooth_err] {layer_name} -> alpha={alpha:.2f} {err_str}")

        smooth_scale = act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
        smooth_scale = smooth_scale.clamp_min(1e-8)

    else:
        # Config mode: no calibration data — apply weight-only smoothing via preset alpha
        alpha = max(getattr(m.weight_quantizer, "smooth_alpha", 0.5), 0.0)
        if alpha > 0:
            weight_absmax = m.weight.detach().abs().amax(dim=0).clamp_min(1e-8)
            smooth_scale = weight_absmax.pow(alpha - 1.0)
        else:
            smooth_scale = None

    # Weight stats + freeze + branch build
    stat_weight = m.weight * \
        smooth_scale.reshape(1, -1) if smooth_scale is not None else m.weight
    m.weight_quantizer.collect_stats(stat_weight)
    m.weight_quantizer.freeze()
    if hasattr(m.weight_quantizer, "build_branch"):
        m.weight_quantizer.build_branch(m.weight, smooth_scale=smooth_scale)
    m.act_quantizer.freeze()

    return smooth_scale


def _print_freeze_summary(search_smooth_alpha, smooth_alpha_override, alpha_stats, module):
    any_override = smooth_alpha_override is not None
    has_any_search = search_smooth_alpha or any_override
    if not has_any_search:
        total_layers = sum(1 for _, m in iter_quant_layers(module))
        print(f"[W4A4] freeze done ({total_layers} layers, fixed alpha)")
    if has_any_search and alpha_stats:
        print("[alpha_search] per-layer results:")
        for al, cnt in sorted(alpha_stats.items()):
            print(f"  alpha={al:.2f} -> {cnt} layers")
        total = sum(alpha_stats.values())
        primary = max(alpha_stats, key=alpha_stats.get)
        print(
            f"  total: {total} layers, most common: alpha={primary:.2f} ({alpha_stats[primary]} layers)")


@torch.no_grad()
def freeze_quant_params(module: nn.Module, search_smooth_alpha: bool = False,
                        compute_error: bool = True, smooth_alpha_override=None,
                        alpha_grid_size: int = 7):
    alpha_stats: dict[float, int] = {}
    alpha_per_layer: list[tuple[str, float]] = []

    for name, m in module.named_modules():
        if not isinstance(m, QuantLinearW4A4):
            continue
        freeze_one_layer(
            m, name,
            search_smooth_alpha=search_smooth_alpha,
            compute_error=compute_error,
            smooth_alpha_override=smooth_alpha_override,
            alpha_stats=alpha_stats,
            alpha_per_layer=alpha_per_layer,
            extract_act_qparams=True,
            alpha_grid_size=alpha_grid_size,
        )

    _print_freeze_summary(search_smooth_alpha,
                          smooth_alpha_override, alpha_stats, module)


@torch.no_grad()
def freeze_quant_params_layer_cascade(
    module: nn.Module,
    calib_data: list,
    cascade_forward_fn: callable,
    num_cascade_calib: int = 4,
    search_smooth_alpha: bool = False,
    compute_error: bool = True,
    smooth_alpha_override=None,
    alpha_grid_size: int = 7,
):
    """Layer-by-layer cascade freeze. Each layer is frozen sequentially so that
    layer N sees realistic (pre-quantized) activations from layers 0..N-1.

    Phase 1: FP16 forward → collect act_absmax for all layers.
    Phase 2: For each quantized layer in execution order:
              reset act_absmax + input_cache → cascade forward (prev layers quantized)
              → search alpha → freeze → enable quant.
    """

    # # Phase 1: FP16 forward for act_absmax collection
    # set_observer_enabled(module, True)
    # set_quant_enabled(module, False)
    # for model_input, timesteps, pooled_prompt_embeds, weight_dtype in calib_data:
    #     cascade_forward_fn(model_input, timesteps,
    #                        pooled_prompt_embeds, weight_dtype)

    # # Clear all input_caches (re-collected per layer in cascade)
    # for _, m in iter_quant_layers(module):
    #     m.weight_quantizer.input_cache = []

    # Phase 2: Layer-by-layer cascade
    set_observer_enabled(module, False)
    all_layers = list(iter_quant_layers(module))

    alpha_stats: dict[float, int] = {}
    alpha_per_layer: list[tuple[str, float]] = []
    cascade_calib_count = min(num_cascade_calib, len(calib_data))

    for layer_idx, (layer_name, m) in enumerate(all_layers):
        # Reset this layer to collect cascaded inputs
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

        # Run cascade forward: all previously frozen layers are quantized,
        # this layer collects input_cache from realistic activations
        for i in range(cascade_calib_count):
            cascade_forward_fn(*calib_data[i])

        # Disable observer
        m.weight_quantizer.observer_enabled = False
        if hasattr(m, "act_quantizer"):
            m.act_quantizer.observer_enabled = False

        freeze_one_layer(
            m, layer_name,
            search_smooth_alpha=search_smooth_alpha,
            compute_error=compute_error,
            smooth_alpha_override=smooth_alpha_override,
            alpha_stats=alpha_stats,
            alpha_per_layer=alpha_per_layer,
            extract_act_qparams=True,
            alpha_grid_size=alpha_grid_size,
        )
        m.weight_quantizer.enabled = True
        m.act_quantizer.enabled = True

    _print_freeze_summary(search_smooth_alpha,
                          smooth_alpha_override, alpha_stats, module)
