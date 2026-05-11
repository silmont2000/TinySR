import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline

sys.path.append(".")

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from train.quant_w4a4 import (
    replace_linear_with_w4a4,
    set_quant_enabled,
    set_observer_enabled,
    freeze_quant_params,
    collect_quant_meta,
)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Clean W4A4 fake-quant inference for TinySR.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="checkpoint/tinybackbone/prune-12-merge-tinysr",
    )
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--input_dir", type=str, default="dataset/test_image/")
    parser.add_argument("--output_dir", type=str, default="outputs/w4a4_tinysr")

    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--svdq_rank", type=int, default=32)
    parser.add_argument("--svdq_quantize_residual", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="adain")

    parser.add_argument(
        "--quant_scope",
        type=str,
        choices=["none", "ffn_only", "attn_only", "dit_full"],
        default="ffn_only",
    )
    parser.add_argument("--calib_images", type=int, default=8)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--save_quant_meta", action="store_true")
    # parser.add_argument("--disable_color_fix_for_calib", action="store_true")

    return parser.parse_args()


def get_weight_dtype(args):
    if args.mixed_precision == "fp16":
        return torch.float16
    return torch.float32


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


def get_image_names(input_dir):
    if os.path.isdir(input_dir):
        image_names = []
        for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
            image_names.extend(glob.glob(os.path.join(input_dir, pattern)))
        return sorted(image_names)

    if os.path.isfile(input_dir):
        return [input_dir]

    raise FileNotFoundError(f"Input path does not exist: {input_dir}")


def load_models(args, device, weight_dtype):
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    )

    vae = AutoencoderTiny.from_pretrained(
        args.vae_path,
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    )

    if args.lora_dir:
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "proj",
                "linear",
                "linear_1",
                "linear_2",
                "net.2",
            ],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()

        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            args.lora_dir,
            weight_name="transformer.safetensors",
            cache_dir=args.cache_dir,
        )
        load_lora_state_dict(transformer_lora_state_dict, transformer)

    transformer = transformer.to(device, dtype=weight_dtype).eval()
    vae = vae.to(device, dtype=weight_dtype).eval()
    return transformer, vae


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


def gaussian_weights(tile_width, tile_height, nbatches, in_channels, device, dtype):
    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    midpoint_y = tile_height / 2

    x = torch.arange(tile_width, device=device, dtype=torch.float32)
    y = torch.arange(tile_height, device=device, dtype=torch.float32)

    x_probs = torch.exp(-((x - midpoint_x) ** 2) / (tile_width * tile_width) / (2 * var))
    y_probs = torch.exp(-((y - midpoint_y) ** 2) / (tile_height * tile_height) / (2 * var))

    weights = torch.outer(y_probs, x_probs)
    weights = weights.to(dtype=dtype)
    return weights.expand(nbatches, in_channels, tile_height, tile_width)


@torch.no_grad()
def tile_sample(
    lq_latent,
    transformer,
    timesteps,
    pooled_prompt_embeds,
    weight_dtype,
    latent_tiled_size=64,
    latent_tiled_overlap=8,
):
    _, _, height, width = lq_latent.size()
    tile_size = latent_tiled_size
    tile_overlap = latent_tiled_overlap

    if height * width <= tile_size * tile_size:
        model_pred = transformer(
            hidden_states=lq_latent,
            timestep=timesteps,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        return model_pred.to(lq_latent.device, dtype=weight_dtype)

    tile_size = min(tile_size, min(height, width))
    tile_weights = gaussian_weights(
        tile_size,
        tile_size,
        1,
        transformer.config.in_channels,
        lq_latent.device,
        weight_dtype,
    )

    grid_rows = 0
    cur_x = 0
    while cur_x < width:
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1

    grid_cols = 0
    cur_y = 0
    while cur_y < height:
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1

    noise_preds = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if row == grid_rows - 1:
                ofs_x = width - tile_size
            else:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)

            if col == grid_cols - 1:
                ofs_y = height - tile_size
            else:
                ofs_y = max(col * tile_size - tile_overlap * col, 0)

            input_tile = lq_latent[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size]
            pred = transformer(
                hidden_states=input_tile.to(lq_latent.device, dtype=weight_dtype),
                timestep=timesteps,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            noise_preds.append(pred)

    noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=weight_dtype)
    contributors = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=weight_dtype)

    for row in range(grid_rows):
        for col in range(grid_cols):
            if row == grid_rows - 1:
                ofs_x = width - tile_size
            else:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)

            if col == grid_cols - 1:
                ofs_y = height - tile_size
            else:
                ofs_y = max(col * tile_size - tile_overlap * col, 0)

            index = row * grid_cols + col
            noise_pred[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += noise_preds[index] * tile_weights
            contributors[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += tile_weights

    model_pred = noise_pred / contributors.clamp_min(1e-8)
    return model_pred.to(lq_latent.device, dtype=weight_dtype)


def image_to_latent(args, vae, image_path, tensor_transform, device, weight_dtype):
    lr = Image.open(image_path).convert("RGB")
    resize_flag, new_width, new_height, ori_width, ori_height = preprocess_one_image(
        lr,
        args.upscale,
        args.process_size,
    )

    lr_scale = lr.resize((int(ori_width * args.upscale), int(ori_height * args.upscale)))

    pixel_values = tensor_transform(lr).unsqueeze(0).to(device=device, dtype=weight_dtype)
    pixel_values = torch.nn.functional.interpolate(
        pixel_values,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
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


@torch.no_grad()
def calibrate_w4a4(
    args,
    transformer,
    vae,
    image_names,
    pooled_prompt_embeds,
    timesteps,
    weight_dtype,
):
    if args.quant_scope == "none":
        return

    calib_count = min(max(args.calib_images, 1), len(image_names))
    calib_names = image_names[:calib_count]
    device = torch.device(args.device)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    # PTQ calibration stage: collect activation statistics only.
    # Keep fake-quant disabled to avoid perturbing downstream tensors.
    set_quant_enabled(transformer, False)
    set_observer_enabled(transformer, True)

    for image_path in tqdm(calib_names, desc="Calibrating W4A4"):
        model_input, _ = image_to_latent(
            args,
            vae,
            image_path,
            tensor_transform,
            device,
            weight_dtype,
        )

        _ = tile_sample(
            model_input,
            transformer,
            timesteps,
            pooled_prompt_embeds,
            weight_dtype,
            latent_tiled_size=args.latent_tiled_size,
            latent_tiled_overlap=args.latent_tiled_overlap,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    freeze_quant_params(transformer)
    set_quant_enabled(transformer, True)
    set_observer_enabled(transformer, False)

    print(f"[W4A4] calibration finished with {calib_count} images.")


@torch.no_grad()
def run_inference(
    args,
    transformer,
    vae,
    image_names,
    pooled_prompt_embeds,
    timesteps,
    weight_dtype,
):
    device = torch.device(args.device)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    os.makedirs(args.output_dir, exist_ok=True)

    warmup_left = max(0, args.warmup_images)
    per_image_time = []

    for image_path in tqdm(image_names, desc="Infer"):
        model_input, image_info = image_to_latent(
            args,
            vae,
            image_path,
            tensor_transform,
            device,
            weight_dtype,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        model_pred = tile_sample(
            model_input,
            transformer,
            timesteps,
            pooled_prompt_embeds,
            weight_dtype,
            latent_tiled_size=args.latent_tiled_size,
            latent_tiled_overlap=args.latent_tiled_overlap,
        )
        latent_stu = model_input - model_pred

        image = vae.decode(
            latent_stu / vae.config.scaling_factor,
            return_dict=False,
        )[0]
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
            image_pil = image_pil.resize(
                (
                    int(image_info["ori_width"] * args.upscale),
                    int(image_info["ori_height"] * args.upscale),
                )
            )

        if args.align_method == "adain":
            image_pil = adain_color_fix(target=image_pil, source=image_info["lr"])
        elif args.align_method == "wavelet":
            image_pil = wavelet_color_fix(target=image_pil, source=image_info["lr_scale"])

        save_path = os.path.join(args.output_dir, os.path.basename(image_path))
        image_pil.save(save_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if per_image_time:
        timing = {
            "avg_time_sec": float(np.mean(per_image_time)),
            "p50_time_sec": float(np.percentile(per_image_time, 50)),
            "p90_time_sec": float(np.percentile(per_image_time, 90)),
            "timed_image_count": len(per_image_time),
            "warmup_images": int(args.warmup_images),
        }
    else:
        timing = {
            "avg_time_sec": None,
            "p50_time_sec": None,
            "p90_time_sec": None,
            "timed_image_count": 0,
            "warmup_images": int(args.warmup_images),
        }

    return timing


def save_report(args, replaced_layers, quant_meta, timing):
    report = {
        "args": vars(args),
        "replaced_layers": replaced_layers,
        "quant_meta": quant_meta,
        "timing": timing,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "w4a4_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] report: {report_path}")


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args)

    image_names = get_image_names(args.input_dir)
    if len(image_names) == 0:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    print(f"[INFO] images: {len(image_names)}")
    print(f"[INFO] quant_scope: {args.quant_scope}")
    print(f"[INFO] output_dir: {args.output_dir}")

    transformer, vae = load_models(args, device, weight_dtype)

    replaced_layers = []
    if args.quant_scope != "none":
        target_suffixes = get_target_suffixes(args.quant_scope)
        replaced_layers = replace_linear_with_w4a4(
            transformer,
            target_suffixes=target_suffixes,
            skip_keywords=("lora_",),
            weight_quant_kind="svdq",
            weight_quant_kwargs={
                "bits": 4,
                "symmetric": True,
                "per_channel": True,
                "ch_axis": 0,
                "rank": args.svdq_rank,
                "compensate": True,
                "quantize_residual": args.svdq_quantize_residual,
            },
        )
        print(f"[W4A4] replaced Linear layers: {len(replaced_layers)}")
        for layer_name in replaced_layers[:20]:
            print(f"  - {layer_name}")
        if len(replaced_layers) > 20:
            print(f"  ... and {len(replaced_layers) - 20} more")

    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"),
        map_location=device,
    ).to(device=device, dtype=weight_dtype)

    timesteps = torch.tensor(
        [args.timestep],
        device=device,
        dtype=weight_dtype,
    )

    calibrate_w4a4(
        args,
        transformer,
        vae,
        image_names,
        pooled_prompt_embeds,
        timesteps,
        weight_dtype,
    )

    quant_meta = collect_quant_meta(transformer) if args.quant_scope != "none" else []

    timing = run_inference(
        args,
        transformer,
        vae,
        image_names,
        pooled_prompt_embeds,
        timesteps,
        weight_dtype,
    )

    print(
        "[TIME] "
        f"avg={timing['avg_time_sec']} "
        f"p50={timing['p50_time_sec']} "
        f"p90={timing['p90_time_sec']} "
        f"count={timing['timed_image_count']}"
    )

    if args.save_quant_meta:
        save_report(args, replaced_layers, quant_meta, timing)
    else:
        save_report(args, replaced_layers, [], timing)


if __name__ == "__main__":
    main()
