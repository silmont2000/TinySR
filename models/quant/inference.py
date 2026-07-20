"""Quantization-specific pipeline: layer replacement, calibration entry point."""
import os
import time

import torch
from torchvision import transforms
from tqdm import tqdm

from utils.device import get_optimal_device_name

from models.quant.layers import (
    get_target_suffixes,
    replace_linear_with_w4a4,
    load_smooth_alpha_from_report,
)
from models.quant.calibrate import run_calibration, save_calib_cache, load_calib_cache
from models.pipeline import (
    get_image_names,
    get_weight_dtype,
    image_to_latent,
    load_models,
    run_inference,
    save_report,
)

# ---- re-exports for backward compat (gradual migration) ------------------
__all__ = [
    "build_layer_replacement_kwargs",
    "replace_quant_layers",
    "calibrate_w4a4",
    "get_image_names",
    "get_weight_dtype",
    "image_to_latent",
    "load_models",
    "run_inference",
    "save_report",
]
# ---- layer replacement utilities ----------------------------------------

def build_layer_replacement_kwargs(w_bits, a_bits, svdq_rank, svdq_smooth_alpha,
                                   svdq_iterations=0, act_group_size=64, weight_group_size=-1,
                                   asymmetric=False):
    symmetric = not asymmetric
    return {
        "weight_quant_kwargs": {
            "bits": w_bits, "symmetric": symmetric, "per_channel": True,
            "ch_axis": 0, "rank": svdq_rank, "smooth_alpha": svdq_smooth_alpha,
            "num_svd_iterations": svdq_iterations,
            "weight_group_size": weight_group_size,
        },
        "act_quant_kwargs": {
            "bits": a_bits, "symmetric": symmetric, "per_channel": False,
            "group_size": act_group_size,
        },
    }


def replace_quant_layers(transformer, quant_scope,
                         w_bits, a_bits, svdq_rank, svdq_smooth_alpha,
                         svdq_iterations=0, act_group_size=64, weight_group_size=-1,
                         ffn_blocks=None, asymmetric=False):
    if quant_scope == "none":
        return []
    target_suffixes = get_target_suffixes(quant_scope, ffn_blocks=ffn_blocks)
    quant_kwargs = build_layer_replacement_kwargs(
        w_bits, a_bits, svdq_rank, svdq_smooth_alpha, svdq_iterations,
        act_group_size, weight_group_size, asymmetric=asymmetric)

    replaced = replace_linear_with_w4a4(
        transformer,
        target_suffixes=target_suffixes,
        skip_keywords=("lora_",),
        weight_quant_kwargs=quant_kwargs["weight_quant_kwargs"],
        act_quant_kwargs=quant_kwargs["act_quant_kwargs"],
    )

    if len(replaced) > 20:
        print(f"  ... and {len(replaced) - 20} more")
    return replaced


# ---- calibration entry-point --------------------------------------------

@torch.no_grad()
def calibrate_w4a4(
    transformer, vae, calib_image_names,
    pooled_prompt_embeds, timesteps, weight_dtype,
    quant_scope, calib_images,
    search_mode,
    cascade_calib_images,
    load_smooth_alpha_report, latent_tiled_size, latent_tiled_overlap,
    device=None, upscale=4, process_size=512,
    alpha_grid_size=7,
    no_smooth=False,
    calib_data_list=None,
    no_gptq=False,
    no_svd_early_stop=False,
):
    """Prepare calibration data and run the calibration pipeline.
    
    Args:
        calib_data_list: Optional pre-built list of (latent, timesteps, embeds, dtype)
            tuples. When provided, skips VAE encoding of calibration images.
            Used by DiTAS data-free mode.
        no_gptq: If True, use minmax instead of GPTQ for residual quantization.
        no_svd_early_stop: If True, run all SVD iterations without early-stopping.
    """
    if quant_scope == "none":
        return
    if device is None:
        device = get_optimal_device_name()
    device = torch.device(device)

    from models.quant.tiler import tile_sample

    if calib_data_list is None:
        tensor_transform = transforms.Compose([transforms.ToTensor()])
        calib_count = min(max(calib_images, 1), len(calib_image_names))
        calib_names = calib_image_names[:calib_count]

        calib_data_list = []
        for image_path in tqdm(calib_names, desc="Building calib data"):
            model_input, _ = image_to_latent(
                upscale, process_size, vae, image_path, tensor_transform, device, weight_dtype)
            calib_data_list.append((model_input, timesteps, pooled_prompt_embeds, weight_dtype))

    def _forward_fn(model_input, ts, ppe, wd):
        tile_sample(model_input, transformer, ts, ppe, wd,
                    latent_tiled_size=latent_tiled_size,
                    latent_tiled_overlap=latent_tiled_overlap)

    smooth_alpha_override = None
    if load_smooth_alpha_report:
        smooth_alpha_override = load_smooth_alpha_from_report(load_smooth_alpha_report)

    run_calibration(
        transformer, calib_data_list, _forward_fn,
        quant_scope=quant_scope,
        search_mode=search_mode,
        cascade_calib_images=cascade_calib_images,
        smooth_alpha_override=smooth_alpha_override,
        alpha_grid_size=alpha_grid_size,
        no_smooth=no_smooth,
        no_gptq=no_gptq,
        no_svd_early_stop=no_svd_early_stop,
    )
