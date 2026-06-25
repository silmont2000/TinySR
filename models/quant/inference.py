import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from utils.device import get_optimal_device_name
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline

from models.quant.layers import (
    get_target_suffixes,
    replace_linear_with_w4a4,
    replace_linear_with_w4a4_from_config,
    load_smooth_alpha_from_report,
)
from models.quant.tiler import gaussian_weights, tile_sample
from models.quant.calibration import calibrate_and_freeze
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from utils.util import load_lora_state_dict
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.tinysr.tinysd3 import TinySD3Transformer2DModel


def get_weight_dtype(mixed_precision):
    if mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
        return torch.bfloat16
    else:
        return torch.float32


def get_image_names(input_dir):
    if os.path.isdir(input_dir):
        image_names = []
        for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
            image_names.extend(glob.glob(os.path.join(input_dir, pattern)))
        return sorted(image_names)
    if os.path.isfile(input_dir):
        return [input_dir]
    raise FileNotFoundError(f"Input path does not exist: {input_dir}")


def preprocess_one_image(lr, upscale, process_size):
    ori_width, ori_height = lr.size
    resize_flag = False
    if ori_width < process_size // upscale or ori_height < process_size // upscale:
        scale = (process_size // upscale) / min(ori_width, ori_height)
        new_width = int(scale * ori_width)
        new_height = int(scale * ori_height)
        resize_flag = True
    else:
        new_width = ori_width
        new_height = ori_height

    new_width = upscale * new_width
    new_height = upscale * new_height

    if new_width % 8 or new_height % 8:
        resize_flag = True
        new_width = new_width - new_width % 8
        new_height = new_height - new_height % 8

    return resize_flag, new_width, new_height, ori_width, ori_height


def image_to_latent(upscale, process_size, vae, image_path, tensor_transform, device, weight_dtype):
    lr = Image.open(image_path).convert("RGB")
    resize_flag, new_width, new_height, ori_width, ori_height = preprocess_one_image(
        lr, upscale, process_size)

    lr_scale = lr.resize((int(ori_width * upscale), int(ori_height * upscale)))

    pixel_values = tensor_transform(lr).unsqueeze(
        0).to(device=device, dtype=weight_dtype)
    pixel_values = torch.nn.functional.interpolate(
        pixel_values, size=(new_height, new_width), mode="bilinear", align_corners=False)
    pixel_values = pixel_values * 2 - 1
    pixel_values = pixel_values.to(device=device, dtype=weight_dtype)

    model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor

    image_info = {
        "lr": lr,
        "lr_scale": lr_scale,
        "resize_flag": resize_flag,
        "ori_width": ori_width,
        "ori_height": ori_height,
    }
    return model_input, image_info


def load_models(pretrained_model_name_or_path, vae_path, lora_dir, rank,
                cache_dir, device, weight_dtype, skip_lora=False):
    transformer = TinySD3Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )

    vae = AutoencoderTiny.from_pretrained(
        vae_path,
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    )

    if lora_dir and not skip_lora:
        transformer_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "proj", "linear", "linear_1", "linear_2", "net.2",
            ],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            lora_dir, weight_name="transformer.safetensors", cache_dir=cache_dir)
        load_lora_state_dict(transformer_lora_state_dict, transformer)

    transformer = transformer.to(device, dtype=weight_dtype).eval()
    vae = vae.to(device, dtype=weight_dtype).eval()
    return transformer, vae


def build_layer_replacement_kwargs(w_bits, a_bits, svdq_rank, svdq_smooth_alpha, svdq_iterations=0, act_group_size=64, weight_group_size=-1):
    return {
        "weight_quant_kwargs": {
            "bits": w_bits, "symmetric": True, "per_channel": True,
            "ch_axis": 0, "rank": svdq_rank, "smooth_alpha": svdq_smooth_alpha,
            "num_svd_iterations": svdq_iterations,
            "weight_group_size": weight_group_size,
        },
        "act_quant_kwargs": {
            "bits": a_bits, "symmetric": True, "per_channel": False,
            "group_size": act_group_size,
        },
    }


def replace_quant_layers(transformer, quant_scope, quant_config,
                         w_bits, a_bits, svdq_rank, svdq_smooth_alpha,
                         svdq_iterations=0, act_group_size=64, weight_group_size=-1):
    if quant_scope == "none":
        return []
    target_suffixes = get_target_suffixes(quant_scope)
    quant_kwargs = build_layer_replacement_kwargs(
        w_bits, a_bits, svdq_rank, svdq_smooth_alpha, svdq_iterations, act_group_size, weight_group_size)

    if quant_config is not None:
        replaced = replace_linear_with_w4a4_from_config(
            transformer, quant_config,
            target_suffixes=target_suffixes,
            skip_keywords=("lora_",),
            default_weight_quant_kind="svdq",
            default_weight_quant_kwargs=quant_kwargs["weight_quant_kwargs"],
            default_act_quant_kwargs=quant_kwargs["act_quant_kwargs"],
        )
    else:
        replaced = replace_linear_with_w4a4(
            transformer,
            target_suffixes=target_suffixes,
            skip_keywords=("lora_",),
            weight_quant_kind="svdq",
            weight_quant_kwargs=quant_kwargs["weight_quant_kwargs"],
            act_quant_kwargs=quant_kwargs["act_quant_kwargs"],
        )

    if len(replaced) > 20:
        print(f"  ... and {len(replaced) - 20} more")
    return replaced


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
):
    """Run calibration with the smooth_alpha priority chain.

    search_mode: None → use svdq_smooth_alpha from layer construction
                 "grid" → single-pass grid search
                 "cascade" → layer-by-layer cascade freeze
    The caller is responsible for setting the correct svdq_smooth_alpha
    in weight_quant_kwargs before calling this function.
    """
    if quant_scope == "none":
        return
    if device is None:
        device = get_optimal_device_name()
    device = torch.device(device)
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    calib_count = min(max(calib_images, 1), len(calib_image_names))
    calib_names = calib_image_names[:calib_count]

    calib_data_list = []
    for image_path in tqdm(calib_names, desc="Building calib data"):
        model_input, _ = image_to_latent(
            upscale, process_size, vae, image_path, tensor_transform, device, weight_dtype)
        calib_data_list.append(
            (model_input, timesteps, pooled_prompt_embeds, weight_dtype))

    def _forward_fn(model_input, ts, ppe, wd):
        tile_sample(model_input, transformer, ts, ppe, wd,
                    latent_tiled_size=latent_tiled_size,
                    latent_tiled_overlap=latent_tiled_overlap)

    smooth_alpha_override = None
    if load_smooth_alpha_report:
        smooth_alpha_override = load_smooth_alpha_from_report(
            load_smooth_alpha_report)

    calibrate_and_freeze(
        transformer, calib_data_list, _forward_fn,
        quant_scope=quant_scope,
        search_mode=search_mode,
        cascade_calib_images=cascade_calib_images,
        smooth_alpha_override=smooth_alpha_override,
        latent_tiled_size=latent_tiled_size,
        latent_tiled_overlap=latent_tiled_overlap,
        alpha_grid_size=alpha_grid_size,
    )


@torch.no_grad()
def run_inference(
    transformer, vae, image_names,
    pooled_prompt_embeds, timesteps, weight_dtype,
    output_dir, upscale, process_size,
    align_method, warmup_images,
    latent_tiled_size, latent_tiled_overlap,
    device=None,
):
    if device is None:
        device = get_optimal_device_name()
    device = torch.device(device)
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    os.makedirs(output_dir, exist_ok=True)

    warmup_left = max(0, warmup_images)
    per_image_time = []

    for image_path in tqdm(image_names, desc="Infer"):
        model_input, image_info = image_to_latent(
            upscale, process_size, vae, image_path, tensor_transform, device, weight_dtype)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        model_pred = tile_sample(
            model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
            latent_tiled_size=latent_tiled_size,
            latent_tiled_overlap=latent_tiled_overlap,
        )
        latent_stu = model_input - model_pred

        image = vae.decode(
            latent_stu / vae.config.scaling_factor, return_dict=False)[0]
        image = image.squeeze(0).clamp(-1, 1)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        if warmup_left > 0:
            warmup_left -= 1
        else:
            per_image_time.append(elapsed)

        image_pil = transforms.ToPILImage()(image.cpu() / 2 + 0.5)

        if image_info["resize_flag"]:
            image_pil = image_pil.resize((
                int(image_info["ori_width"] * upscale),
                int(image_info["ori_height"] * upscale),
            ))

        if align_method == "adain":
            image_pil = adain_color_fix(
                target=image_pil, source=image_info["lr"])
        elif align_method == "wavelet":
            image_pil = wavelet_color_fix(
                target=image_pil, source=image_info["lr_scale"])

        save_path = os.path.join(output_dir, os.path.basename(image_path))
        image_pil.save(save_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return _build_timing_report(per_image_time, warmup_images)


def _build_timing_report(per_image_time, warmup_images):
    if per_image_time:
        return {
            "avg_time_sec": float(np.mean(per_image_time)),
            "p50_time_sec": float(np.percentile(per_image_time, 50)),
            "p90_time_sec": float(np.percentile(per_image_time, 90)),
            "timed_image_count": len(per_image_time),
            "warmup_images": int(warmup_images),
        }
    return {
        "avg_time_sec": None, "p50_time_sec": None, "p90_time_sec": None,
        "timed_image_count": 0, "warmup_images": int(warmup_images),
    }


def save_report(output_dir, args_dict, replaced_layers, quant_meta, timing):
    report = {
        "args": args_dict,
        "replaced_layers": replaced_layers,
        "quant_meta": quant_meta,
        "timing": timing,
    }
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "w4a4_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[DONE] report: {report_path}")
