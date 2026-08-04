#!/usr/bin/env python3
"""Dump per-channel mean of every nn.Linear output for TinySR transformer.

Loads a merged_backbone transformer weight (config.json + safetensors) plus a
VAE, runs every image in a folder through the transformer, and records the
per-channel mean of each Linear layer's output (mean over the token dimension,
keeping the output-channel dimension).

Result is saved as a .pt dict:
    {
        "layers":   [layer_name, ...],
        "images":   [image_path, ...],
        "means":    {layer_name: FloatTensor(num_images, C_out)},
    }

Usage:
    python script/dump_layer_means.py \\
        --pretrained "/path/to/merged_backbone" \\
        --vae_path checkpoint/vae/separable \\
        --input_dir /path/to/images \\
        --output outputs/layer_means.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
from tqdm import tqdm

# Add TinySR root to path so `models`/`utils` import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.device import get_optimal_device_name
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.pipeline import get_image_names, image_to_latent


DEFAULT_BACKBONE = (
    "/Users/xieboyang/Documents/Tinysr/可以保留的数据/好/"
    "sweep_tier-2-3/sweep_tier-2-3/merged_backbone"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump per-channel mean of all Linear outputs")
    parser.add_argument("--pretrained", type=str, default=DEFAULT_BACKBONE,
                        help="Path to merged_backbone (config.json + safetensors)")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable",
                        help="Path to VAE weights")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Folder of input LR images")
    parser.add_argument("--output", "-o", type=str, default="outputs/layer_means.pt",
                        help="Output .pt path")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Cap number of images (0 = all)")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default=get_optimal_device_name())
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def get_weight_dtype(mixed_precision, device):
    if mixed_precision == "fp16":
        # fp16 on CPU is unstable/slow; fall back to fp32.
        return torch.float16 if device.type == "cuda" else torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


class PerChannelMeanHook:
    """Accumulate per-channel mean of a module's output for the current image."""

    def __init__(self):
        self.value = None

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        # output: (..., C_out) -> mean over every dim but the last.
        out = output.detach().float().reshape(-1, output.shape[-1])
        self.value = out.mean(dim=0).cpu()


def main():
    args = parse_args()
    device = torch.device(args.device if (
        args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    weight_dtype = get_weight_dtype(args.mixed_precision, device)

    # ── load transformer (merged_backbone has no `transformer` subfolder) ──
    print(f"[load] transformer <- {args.pretrained}")
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    ).to(device, dtype=weight_dtype).eval()

    print(f"[load] vae <- {args.vae_path}")
    vae = AutoencoderTiny.from_pretrained(
        args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir,
    ).to(device, dtype=weight_dtype).eval()

    # ── register hooks on every nn.Linear ─────────────────────
    hooks = {}
    handles = []
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            h = PerChannelMeanHook()
            handles.append(module.register_forward_hook(h))
            hooks[name] = h
    layer_names = sorted(hooks.keys())
    print(f"[hooks] {len(layer_names)} Linear layers")

    # ── resolve images ────────────────────────────────────────
    image_names = get_image_names(args.input_dir)
    if args.max_images > 0:
        image_names = image_names[:args.max_images]
    if not image_names:
        raise SystemExit(f"No images found in {args.input_dir}")
    print(f"[images] {len(image_names)}")

    tensor_transform = transforms.Compose([transforms.ToTensor()])
    pooled_proj = torch.zeros(
        1, transformer.config.pooled_projection_dim,
        device=device, dtype=weight_dtype)
    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)

    # ── per-image, per-layer accumulation ─────────────────────
    means = {name: [] for name in layer_names}

    with torch.no_grad():
        for image_path in tqdm(image_names, desc="[dump]"):
            model_input, _ = image_to_latent(
                args.upscale, args.process_size, vae, image_path,
                tensor_transform, device, weight_dtype)
            model_input = model_input.to(device=device, dtype=weight_dtype)

            for h in hooks.values():
                h.value = None

            transformer(
                hidden_states=model_input,
                timestep=timesteps,
                pooled_projections=pooled_proj,
                return_dict=False,
            )

            for name in layer_names:
                v = hooks[name].value
                means[name].append(
                    v if v is not None
                    else torch.full((0,), float("nan")))

    for handle in handles:
        handle.remove()

    # ── stack into (num_images + 1, C_out) per layer ──────────
    # Extra last row "__mean__" is the mean over all real images (a virtual image).
    stacked = {}
    for name in layer_names:
        vals = [v for v in means[name] if v.numel() > 0]
        if not vals:
            print(f"  [WARN] {name}: no activations captured")
            continue
        if len(vals) != len(image_names):
            print(f"  [WARN] {name}: captured {len(vals)}/{len(image_names)} images")
        per_image = torch.stack(vals, dim=0)
        virtual = per_image.mean(dim=0, keepdim=True)
        stacked[name] = torch.cat([per_image, virtual], dim=0)

    payload = {
        "layers": [n for n in layer_names if n in stacked],
        "images": image_names + ["__mean__"],
        "means": stacked,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[done] {len(stacked)} layers x {len(image_names)} images -> {out_path}")


if __name__ == "__main__":
    main()
