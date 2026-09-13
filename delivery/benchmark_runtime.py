#!/usr/bin/env python
"""Benchmark the self-contained TinySR W4A4 runtime checkpoint.

Protocol mirrors local test/test_time.py:
  - random input tensor (inference_iterations, batch_size, 3, 128, 128), fp16
  - bicubic upsample to process_size x process_size (default 512)
  - one warmup batch + torch.cuda.synchronize()
  - 100 timed iterations (default), full pipeline:
    VAE encode -> transformer -> latent subtract -> VAE decode
  - per-iteration reset_peak_memory_stats, average ms/sample and peak MB

Usage:
    python delivery/benchmark_runtime.py \
        --model_dir delivery/tinysr_w4a4_runtime \
        --batch_size 1 --inference_iterations 100
    python delivery/benchmark_runtime.py \
        --model_dir delivery/tinysr_w4a4_runtime \
        --batch_size 64 --inference_iterations 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
DELIVERY_DIR = os.path.dirname(os.path.abspath(__file__))
if DELIVERY_DIR not in sys.path:
    sys.path.insert(0, DELIVERY_DIR)

from models.vae.autoencoder_tiny import AutoencoderTiny
from runtime_loader import build_runtime_transformer, load_runtime_state


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=str,
                        default="delivery/tinysr_w4a4_runtime")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--cache_dir", type=str,
                        default="/root/autodl-tmp/hf_cache")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str,
                        choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--inference_iterations", type=int, default=100)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional JSON file to save the benchmark summary.")
    return parser.parse_args()


def main(args, pixel_values, size, vae, transformer,
         timesteps, pooled_prompt_embeds, weight_dtype, device):
    with torch.no_grad():
        pixel_values = torch.nn.functional.interpolate(
            pixel_values, size=size, mode="bicubic", align_corners=False)
        pixel_values = pixel_values * 2 - 1
        pixel_values = pixel_values.to(device, dtype=weight_dtype).clamp(-1, 1)

        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        model_input = model_input.to(device, dtype=weight_dtype)

        model_pred = transformer(
            hidden_states=model_input,
            timestep=timesteps,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        denoised = model_input - model_pred

        image = vae.decode(
            denoised / vae.config.scaling_factor,
            return_dict=False,
        )[0].clamp(-1, 1)
        return image


def run():
    args = parse_args()
    os.chdir(REPO_ROOT)
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HF_HUB_CACHE"] = args.cache_dir

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[BENCH] loading runtime checkpoint from {args.model_dir} ...")
    transformer, _ = build_runtime_transformer(args.model_dir, weight_dtype)
    load_runtime_state(transformer, args.model_dir, weight_dtype)
    transformer = transformer.to(device=device, dtype=weight_dtype).eval()

    print("[BENCH] loading VAE ...")
    vae = AutoencoderTiny.from_pretrained(
        args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)
    vae = vae.to(device=device, dtype=weight_dtype).eval()

    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"),
        map_location=device,
    ).to(dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.expand(args.batch_size, -1)

    size = (args.process_size, args.process_size)
    pixel_values = torch.randn(
        args.inference_iterations, args.batch_size, 3, 128, 128,
        dtype=weight_dtype, device=device,
    )

    print("[BENCH] warmup 1 iteration ...")
    _ = main(args, pixel_values[0], size, vae, transformer,
             timesteps, pooled_prompt_embeds, weight_dtype, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = 0.0
    mem_records = []
    for pixel_value in tqdm(pixel_values, desc="Inference"):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        _ = main(args, pixel_value, size, vae, transformer,
                 timesteps, pooled_prompt_embeds, weight_dtype, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_records.append(torch.cuda.max_memory_allocated() / 1024**2)
        total_time += time.time() - start_time

    avg_sec = total_time / args.inference_iterations / args.batch_size
    avg_ms = avg_sec * 1000.0
    summary = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size,
        "inference_iterations": args.inference_iterations,
        "input": "128x128 -> 512x512",
        "mixed_precision": args.mixed_precision,
        "avg_sec_per_sample": avg_sec,
        "avg_ms_per_sample": avg_ms,
        "total_sec": total_time,
    }
    if mem_records:
        mem_arr = np.array(mem_records)
        summary["peak_mem_avg_mb"] = float(np.mean(mem_arr))
        summary["peak_mem_max_mb"] = float(np.max(mem_arr))
        print(f"Average time: {avg_sec:.6f} sec/sample ({avg_ms:.2f} ms/sample)")
        print(f"Peak mem  avg: {np.mean(mem_arr):.0f} MB")
        print(f"Peak mem  max: {np.max(mem_arr):.0f} MB")
    else:
        print(f"Average time: {avg_sec:.6f} sec/sample ({avg_ms:.2f} ms/sample)")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[BENCH] summary -> {args.output_json}")


if __name__ == "__main__":
    run()
