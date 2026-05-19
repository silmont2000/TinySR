# fmt:off
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from models.quant.inference import (
    get_image_names,
    get_weight_dtype,
    load_models,
    replace_quant_layers,
    image_to_latent,
    tile_sample,
)
from models.quant.layers import set_quant_enabled, set_observer_enabled
from models.vae.autoencoder_tiny import AutoencoderTiny


def parse_args():
    parser = argparse.ArgumentParser(description="W4A4 quantized inference timing benchmark.")

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--input_dir", type=str, default="dataset/StableSR_testsets/DIV2K_V2_val/test_LR")

    parser.add_argument("--load_quant_state", type=str, required=True,
                        help="Path to a previously saved quantized state_dict (.pt). "
                             "Produced by --save_quant_state during calibration.")
    parser.add_argument("--quant_scope", type=str, choices=["ffn_only", "attn_only", "dit_full"],
                        default="dit_full")
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--svdq_rank", type=int, default=32)
    parser.add_argument("--svdq_smooth_alpha", type=float, default=0.5)

    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations before timing.")
    parser.add_argument("--num_images", type=int, default=0,
                        help="Number of images to time (0 = all).")
    parser.add_argument("--skip_decode", action="store_true",
                        help="Skip VAE decode in timing (measure forward-only).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save decoded images to this directory (I/O excluded from timing).")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank (must match calibration).")

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.mixed_precision)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    print(f"[bench] device={args.device}  dtype={args.mixed_precision}  scope={args.quant_scope}")
    print(f"[bench] loading model ...")
    transformer, vae = load_models(
        args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
        args.rank, args.cache_dir, device, weight_dtype, skip_lora=False)

    print(f"[bench] replacing Linear → QuantLinearW4A4 ...")
    replaced = replace_quant_layers(
        transformer, args.quant_scope, None,
        args.w_bits, args.a_bits, args.svdq_rank, args.svdq_smooth_alpha)
    print(f"[bench]   {len(replaced)} layers replaced")

    print(f"[bench] loading state_dict <- {args.load_quant_state}")
    state = torch.load(args.load_quant_state, map_location="cpu")
    # Load state with strict=False first to handle shape-compatible keys.
    # Then directly walk the module tree to handle shape-changed keys
    # (quantizer scales go from scalar→per-channel after calibration).
    model_sd = transformer.state_dict()
    matched = {k: v for k, v in state.items() if k in model_sd and v.shape == model_sd[k].shape}
    missing, unexpected = transformer.load_state_dict(matched, strict=False)

    # --- Handle shape-mismatched keys by walking module tree ---
    reshaped = 0
    not_found = 0
    for k, v in state.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            continue  # already loaded above
        if k not in model_sd:
            not_found += 1
            continue
        # Navigate to the owning module and replace the parameter/buffer
        target = transformer
        *module_path, attr_name = k.rsplit('.', 1) if '.' in k else ([], k)
        if module_path:
            try:
                for part in module_path[0].split('.'):
                    target = getattr(target, part)
            except AttributeError:
                not_found += 1
                continue
        # Update parameter or buffer in place
        v = v.to(device=model_sd[k].device, dtype=model_sd[k].dtype)
        for pname, param in target.named_parameters(recurse=False):
            if pname == attr_name:
                param.data = v
                reshaped += 1
                break
        else:
            for bname, buf in target.named_buffers(recurse=False):
                if bname == attr_name:
                    target.register_buffer(attr_name, v)
                    reshaped += 1
                    break
    if reshaped:
        print(f"[bench]   shape-changed (direct assign): {reshaped}")
    if not_found:
        print(f"[bench]   not in model (skipped): {not_found}")
    if missing:
        print(f"[bench]   missing keys (backbone, expected): {len(missing)}")
    if unexpected:
        print(f"[bench]   unexpected keys (saved but not in model): {len(unexpected)}")

    set_observer_enabled(transformer, False)
    set_quant_enabled(transformer, True)

    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location="cpu",
    ).to(device=device, dtype=weight_dtype)
    timesteps = torch.tensor([args.timestep], device=device, dtype=weight_dtype)

    image_names = get_image_names(args.input_dir)
    if len(image_names) == 0:
        raise RuntimeError(f"No images found in {args.input_dir}")
    print(f"[bench] found {len(image_names)} images")

    # Select images for timing
    n_warmup = max(0, args.warmup)
    n_timed = args.num_images if args.num_images > 0 else max(len(image_names) - n_warmup, 0)
    total_needed = n_warmup + n_timed
    if total_needed > len(image_names):
        n_timed = len(image_names) - n_warmup
        total_needed = len(image_names)

    used = image_names[:total_needed]
    print(f"[bench] warmup={n_warmup}  timed={n_timed}  images={len(used)}/{len(image_names)}")

    param_cnt = sum(p.numel() for p in transformer.parameters())
    print(f"[bench] #Param. {param_cnt/1e6:.1f}M")
    # ---- Output dir ----
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/timing_w{args.w_bits}a{args.a_bits}_r{args.svdq_rank}_{args.quant_scope}_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[bench] output -> {args.output_dir}")
    # ---- Warmup ----
    if n_warmup > 0:
        print("[bench] warming up ...")
        for i, image_path in enumerate(tqdm(used[:n_warmup], desc="Warmup"), 1):
            model_input, image_info = image_to_latent(
                args.upscale, args.process_size, vae, image_path,
                tensor_transform, device, weight_dtype)
            _ = tile_sample(
                model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap)
            if not args.skip_decode:
                latent_stu = model_input - _
                decoded = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0]
            else:
                decoded = None
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- Timing ----
    print(f"[bench] timing {n_timed} images ...")
    times = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    for image_path in tqdm(used[:n_timed], desc="Timing"):
        model_input, _ = image_to_latent(
            args.upscale, args.process_size, vae, image_path,
            tensor_transform, device, weight_dtype)

        if device.type == "cuda":
            torch.cuda.synchronize()
            starter.record()
        else:
            t0 = time.perf_counter()

        model_pred = tile_sample(
            model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
            latent_tiled_size=args.latent_tiled_size,
            latent_tiled_overlap=args.latent_tiled_overlap)

        if not args.skip_decode:
            latent_stu = model_input - model_pred
            decoded = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0]

        if device.type == "cuda":
            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender) / 1000.0
        else:
            elapsed = time.perf_counter() - t0

        times.append(elapsed)
        # Save decoded image AFTER timing (I/O excluded from measurement)
        if decoded is not None and args.output_dir:
            image = decoded.squeeze(0).clamp(-1, 1)
            image_pil = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
            if image_info["resize_flag"]:
                image_pil = image_pil.resize((
                    int(image_info["ori_width"] * args.upscale),
                    int(image_info["ori_height"] * args.upscale)))
            save_path = os.path.join(args.output_dir, os.path.basename(image_path))
            image_pil.save(save_path)


    # ---- Report ----
    if not times:
        print("[bench] no timed images — nothing to report")
        return

    arr = np.array(times)
    print()
    print("=" * 55)
    print(f"  images timed      : {len(times)}")
    print(f"  warmup            : {n_warmup}")
    print(f"  quant_scope       : {args.quant_scope}")
    print(f"  w{args.w_bits}a{args.a_bits} r={args.svdq_rank}")
    print(f"  tiled={args.latent_tiled_size},{args.latent_tiled_overlap}")
    print(f"  decode            : {'skipped' if args.skip_decode else 'included'}")
    print("-" * 55)
    print(f"  avg  (ms)         : {np.mean(arr) * 1000:.1f}")
    print(f"  p50  (ms)         : {np.percentile(arr, 50) * 1000:.1f}")
    print(f"  p90  (ms)         : {np.percentile(arr, 90) * 1000:.1f}")
    print(f"  p95  (ms)         : {np.percentile(arr, 95) * 1000:.1f}")
    print(f"  min  (ms)         : {np.min(arr) * 1000:.1f}")
    print(f"  max  (ms)         : {np.max(arr) * 1000:.1f}")
    print(f"  std  (ms)         : {np.std(arr) * 1000:.1f}")
    print("=" * 55)


if __name__ == "__main__":
    main()