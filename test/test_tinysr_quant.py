"""
Quantized TinySR inference script based on `test/test_tinysr.py`.

It restores quantized weights from `outputs/quant_state.pt` by:
1. loading the original TinySR backbone;
2. attaching LoRA modules when requested;
3. replacing target Linear layers with quantized wrappers;
4. restoring the saved quantized state dict.
"""

import argparse
import glob
import os
import sys
import time

from PIL import Image

import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm

sys.path.append(".")

from models.quant.layers import get_target_suffixes
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.tinysr.tinysd3_nunchaku_w4a4 import (
    NunchakuSVDQW4A4Linear,
    load_nunchaku_svdq_linear_state,
    replace_linear_with_nunchaku_w4a4,
    reset_nunchaku_profile,
    set_nunchaku_profile_enabled,
    summarize_nunchaku_profile,
)
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
from utils.vaehook import _init_tiled_vae
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="checkpoint/tinybackbone/prune-12-merge-tinysr",
        help="Path to the pretrained TinySR transformer backbone.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="checkpoint/vae/separable",
        help="Path to the TinyVAE checkpoint.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="checkpoint/tinysr",
        help="Path to TinySR LoRA weights.",
    )
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/tinysr_quant/")
    parser.add_argument("--input_dir", "-i", type=str, default="dataset/test_image/")
    parser.add_argument("--quant_state_path", type=str, default="outputs/quant_state.pt")
    parser.add_argument("--profile_nunchaku", action="store_true")

    parser.add_argument("--rank", type=int, default=64, help="LoRA rank for transformer.")
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--svdq_rank", type=int, default=32)
    parser.add_argument("--svdq_smooth_alpha", type=float, default=0.55)
    parser.add_argument(
        "--quant_scope",
        type=str,
        choices=["none", "ffn_only", "attn_only", "dit_full"],
        default="dit_full",
    )
    parser.add_argument(
        "--quant_exclude_keywords",
        type=str,
        default="",
        help="Comma-separated layer-name keywords to skip during quantization.",
    )
    parser.add_argument("--is_use_tile", type=bool, default=False)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--align_method",
        type=str,
        choices=["wavelet", "adain", "nofix"],
        default="adain",
    )
    return parser.parse_args()


def _gaussian_weights(tile_width, tile_height, nbatches, device, in_channels):
    from numpy import exp, pi, sqrt
    import numpy as np

    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    midpoint = (latent_width - 1) / 2
    x_probs = [
        exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var)
        for x in range(latent_width)
    ]
    midpoint = latent_height / 2
    y_probs = [
        exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var)
        for y in range(latent_height)
    ]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device), (nbatches, in_channels, 1, 1))


def tile_sample(lq_latent, lq, transformer, timesteps, pooled_prompt_embeds, args):
    with torch.no_grad():
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (args.latent_tiled_size, args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred = transformer(
                hidden_states=lq_latent,
                timestep=timesteps,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = _gaussian_weights(tile_size, tile_size, 1, args.device, transformer.config.in_channels)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols - 1:
                        input_list_t = torch.cat(input_list, dim=0)
                        model_out = transformer(
                            hidden_states=input_list_t,
                            timestep=timesteps,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                        input_list = []

                    noise_preds.append(model_out)

            noise_pred = torch.zeros(lq_latent.shape, device=args.device)
            contributors = torch.zeros(lq_latent.shape, device=args.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    tile_idx = row * grid_cols + col
                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += (
                        noise_preds[tile_idx] * tile_weights
                    )
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            noise_pred /= contributors.clamp_min(1e-8)
            model_pred = noise_pred

    return model_pred


tensor_transforms = transforms.Compose([transforms.ToTensor()])


def load_quant_state(transformer, state_path):
    print(f"[NUNCHAKU] loading quantized state <- {state_path}")
    state = torch.load(state_path, map_location="cpu")

    loaded = 0
    missing = []

    for name, module in transformer.named_modules():
        if not isinstance(module, NunchakuSVDQW4A4Linear):
            continue
        if not load_nunchaku_svdq_linear_state(module, state, name):
            missing.append(f"{name}.weight_quantizer.residual")
            continue
        loaded += 1

    print(f"[NUNCHAKU]   loaded modules: {loaded}")
    if missing:
        print(f"[NUNCHAKU]   missing residual keys: {len(missing)}")


def print_nunchaku_profile(transformer):
    stats = summarize_nunchaku_profile(transformer)
    print("[NUNCHAKU][PROFILE] layers:", stats["layers"])
    print("[NUNCHAKU][PROFILE] called_layers:", stats["called_layers"])
    print("[NUNCHAKU][PROFILE] total_calls:", stats["calls"])
    print("[NUNCHAKU][PROFILE] backend_calls:", stats["backend_calls"])
    print("[NUNCHAKU][PROFILE] fallback_calls:", stats["fallback_calls"])
    print("[NUNCHAKU][PROFILE] cpu_calls:", stats["cpu_calls"])
    print("[NUNCHAKU][PROFILE] low_rank_calls:", stats["low_rank_calls"])
    print("[NUNCHAKU][PROFILE] backend_ms:", round(stats["backend_ms"], 3))
    print("[NUNCHAKU][PROFILE] fallback_ms:", round(stats["fallback_ms"], 3))
    print("[NUNCHAKU][PROFILE] low_rank_ms:", round(stats["low_rank_ms"], 3))
    print("[NUNCHAKU][PROFILE] exception_count:", stats["exception_count"])
    if stats["calls"]:
        backend_ratio = stats["backend_calls"] / stats["calls"]
        fallback_ratio = stats["fallback_calls"] / stats["calls"]
        print("[NUNCHAKU][PROFILE] backend_ratio:", round(backend_ratio, 4))
        print("[NUNCHAKU][PROFILE] fallback_ratio:", round(fallback_ratio, 4))
    if stats["backend_calls"]:
        print(
            "[NUNCHAKU][PROFILE] avg_backend_ms:",
            round(stats["backend_ms"] / stats["backend_calls"], 4),
        )
    if stats["fallback_calls"]:
        print(
            "[NUNCHAKU][PROFILE] avg_fallback_ms:",
            round(stats["fallback_ms"] / stats["fallback_calls"], 4),
        )
    if stats["low_rank_calls"]:
        print(
            "[NUNCHAKU][PROFILE] avg_low_rank_ms:",
            round(stats["low_rank_ms"] / stats["low_rank_calls"], 4),
        )
    if "first_exception_type" in stats:
        print("[NUNCHAKU][PROFILE] first_exception_layer:", stats["first_exception_layer"])
        print("[NUNCHAKU][PROFILE] first_exception_type:", stats["first_exception_type"])
        print("[NUNCHAKU][PROFILE] first_exception_message:", stats["first_exception_message"])


def main_one(args, pixel_values, size, transformer, vae, timesteps, pooled_prompt_embeds):
    with torch.no_grad():
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode="bicubic", align_corners=False)
        pixel_values = pixel_values * 2 - 1

        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor

        model_pred = tile_sample(
            model_input, pixel_values, transformer, timesteps, pooled_prompt_embeds, args
        )
        latent_stu = model_input - model_pred
        image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1, 1)
        return image


if __name__ == "__main__":
    args = parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    )
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)

    if args.is_use_tile:
        _init_tiled_vae(
            vae,
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size,
        )

    if args.lora_dir:
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()

        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            args.lora_dir,
            weight_name="transformer.safetensors",
            cache_dir=args.cache_dir,
        )
        load_lora_state_dict(transformer_lora_state_dict, transformer)

        print("[LoRA] before merge - checking for LoRA parameters:")
        lora_param_count_before = sum(1 for name, _ in transformer.named_parameters() if "lora_" in name.lower())
        print(f"[LoRA]   LoRA-specific parameters before merge: {lora_param_count_before}")
        
        # transformer = transformer.merge_and_unload()
        print("[LoRA] merged LoRA weights into transformer")
        
        print("[LoRA] after merge - verifying LoRA parameters:")
        lora_param_count_after = sum(1 for name, _ in transformer.named_parameters() if "lora_" in name.lower())
        print(f"[LoRA]   LoRA-specific parameters after merge: {lora_param_count_after}")
        print(f"[LoRA]   hasattr 'lora_A': {hasattr(transformer, 'lora_A')}")
        print(f"[LoRA]   hasattr 'active_adapters': {hasattr(transformer, 'active_adapters')}")
        
        if lora_param_count_after == 0:
            print("[LoRA] ✓ LoRA successfully merged and removed from model")
        else:
            print("[LoRA] ⚠ Warning: Some LoRA parameters remain after merge")

    target_suffixes = get_target_suffixes(args.quant_scope)
    exclude_keywords = tuple(s.strip() for s in args.quant_exclude_keywords.split(",") if s.strip())
    replaced = replace_linear_with_nunchaku_w4a4(
        transformer,
        rank=args.svdq_rank,
        target_suffixes=target_suffixes,
        exclude_keywords=exclude_keywords,
    )
    print(f"[NUNCHAKU] replaced layers: {len(replaced)}")
    print(f"[NUNCHAKU] backend: nunchaku SVDQW4A4 (layers={len(replaced)}, rank={args.svdq_rank})")

    transformer = transformer.to(args.device, dtype=weight_dtype)
    vae = vae.to(args.device, dtype=weight_dtype)
    
    # load_quant_state(transformer, args.quant_state_path)
    profiled_layers = set_nunchaku_profile_enabled(transformer, args.profile_nunchaku)
    if args.profile_nunchaku:
        reset_nunchaku_profile(transformer)
        print(f"[NUNCHAKU][PROFILE] enabled on layers: {profiled_layers}")

    param_cnt = sum(p.numel() for p in transformer.transformer_blocks.parameters())
    print("#Param.", param_cnt / 1e6, "M")

    timesteps = torch.tensor([1000.0], device=args.device, dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"),
        map_location=args.device,
    ).to(dtype=weight_dtype)

    if os.path.isdir(args.input_dir):
        image_names = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            image_names.extend(glob.glob(os.path.join(args.input_dir, pattern)))
        image_names = sorted(image_names)
    else:
        image_names = [args.input_dir]

    datalen = len(image_names)
    print("image_num", datalen)
    os.makedirs(args.output_dir, exist_ok=True)

    total_time = 0.0
    for image_name in tqdm(image_names):
        lr = Image.open(image_name).convert("RGB")
        ori_width, ori_height = lr.size
        upscale = args.upscale
        process_size = args.process_size

        resize_flag = False
        if ori_width < process_size // upscale or ori_height < process_size // upscale:
            scale = (process_size // upscale) / min(ori_width, ori_height)
            new_width, new_height = int(scale * ori_width), int(scale * ori_height)
            resize_flag = True
        else:
            new_width, new_height = ori_width, ori_height

        new_width, new_height = upscale * new_width, upscale * new_height
        if new_width % 8 or new_height % 8:
            resize_flag = True
            new_width = new_width - new_width % 8
            new_height = new_height - new_height % 8

        lr_scale = lr.resize((int(ori_width * args.upscale), int(ori_height * args.upscale)))
        pixel_values = tensor_transforms(lr).unsqueeze(0).to(args.device, dtype=weight_dtype)

        start_time = time.time()
        image = main_one(
            args,
            pixel_values,
            (new_height, new_width),
            transformer,
            vae,
            timesteps,
            pooled_prompt_embeds,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()

        image_pil = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
        total_time += end_time - start_time

        if resize_flag:
            image_pil = image_pil.resize((int(ori_width * args.upscale), int(ori_height * args.upscale)))

        if args.align_method == "adain":
            image_pil = adain_color_fix(target=image_pil, source=lr)
        elif args.align_method == "wavelet":
            image_pil = wavelet_color_fix(target=image_pil, source=lr_scale)

        image_pil.save(os.path.join(args.output_dir, os.path.basename(image_name)))
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"Average time: {total_time / max(datalen, 1)}")
    if args.profile_nunchaku:
        print_nunchaku_profile(transformer)