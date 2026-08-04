"""
Compare FP16 vs post-quant activation distributions for deep layers.

Collects act_absmax from:
  1. Phase 1 (all-FP16 forward, same as Grid)
  2. Cascade forward (previous layers quantized, current layer observer-enabled)

Usage:
    python test/test_act_distribution_shift.py \
        --cascade_report outputs/w4a4_svdq_r32_attn_only_acascade_20260714_161136/w4a4_report.json \
        --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr \
        --vae_path checkpoint/vae/separable \
        --lora_dir checkpoint/tinysr \
        --calib_input_dir ../testset/RealSR/LR \
        --calib_images 100 \
        --quant_scope attn_only --quant_ffn_blocks "1,6,9,10,11" \
        --rank 64
"""
import argparse
import gc
import json
import os
import sys

import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.layers import (
    QuantLinearW4A4,
    parse_ffn_blocks,
    set_observer_enabled,
    set_quant_enabled,
)
from models.quant.inference import replace_quant_layers
from models.quant.calibrate import compute_smooth_scale
from models.quant.tiler import tile_sample
from models.pipeline import image_to_latent, get_image_names
from utils.util import load_lora_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade_report", type=str, required=True,
                        help="Path to w4a4_report.json from cascade run")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--calib_input_dir", type=str, default="../testset/RealSR/LR")
    parser.add_argument("--calib_images", type=int, default=100)
    parser.add_argument("--quant_scope", type=str, choices=["ffn_only", "attn_only", "dit_full"], default="attn_only")
    parser.add_argument("--quant_ffn_blocks", type=str, default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_quant_model(transformer, report, device, weight_dtype):
    """Apply cascade alpha to each layer, then freeze (quant enabled)."""
    quant_meta = {m["name"]: m for m in report["quant_meta"]}

    for name, m in transformer.named_modules():
        if not isinstance(m, QuantLinearW4A4):
            continue
        if name not in quant_meta:
            continue

        wq = m.weight_quantizer
        meta = quant_meta[name]["weight"]
        alpha = meta["smooth_alpha"]

        # Compute smooth_scale and apply (same as calibrate_one_layer)
        if wq.act_absmax is not None:
            act_absmax = wq.act_absmax.to(device=m.weight.device, dtype=m.weight.dtype).clamp_min(1e-8)
            w_absmax = m.weight.detach().abs().amax(dim=0).clamp_min(1e-8)
            smooth_scale = compute_smooth_scale(act_absmax, w_absmax, alpha)
            wq.smooth_scale = smooth_scale

        wq.smooth_alpha = alpha
        wq.build_branch(m.weight, smooth_scale=wq.smooth_scale)
        wq.freeze()
        wq.enabled = True

        if hasattr(m, "act_quantizer"):
            m.act_quantizer.freeze()
            m.act_quantizer.enabled = True

    return transformer


def collect_act_stats(transformer, vae, calib_names, pooled_prompt_embeds, timesteps,
                      weight_dtype, device, upscale, process_size, n_images):
    """Collect per-layer act_absmax from forward passes."""
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    # Enable observers on all quant layers
    for m in transformer.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.observer_enabled = True
            if hasattr(m, "act_quantizer"):
                m.act_quantizer.observer_enabled = True

    for image_path in tqdm(calib_names[:n_images], desc="Collecting act stats"):
        model_input, _ = image_to_latent(
            upscale, process_size, vae, image_path, tensor_transform, device, weight_dtype)
        tile_sample(model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                    latent_tiled_size=64, latent_tiled_overlap=8)

    # Collect results
    stats = {}
    for name, m in transformer.named_modules():
        if not isinstance(m, QuantLinearW4A4):
            continue
        wq = m.weight_quantizer
        if wq.act_absmax is not None:
            stats[name] = wq.act_absmax.detach().cpu()

    # Disable observers
    for m in transformer.modules():
        if isinstance(m, QuantLinearW4A4):
            m.weight_quantizer.observer_enabled = False
            if hasattr(m, "act_quantizer"):
                m.act_quantizer.observer_enabled = False

    return stats


def main():
    args = parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load cascade report
    print(f"Loading cascade report: {args.cascade_report}")
    with open(args.cascade_report) as f:
        cascade_report = json.load(f)

    # Load backbone (need two copies)
    print("Loading backbone (FP16 reference)...")
    fp_transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)

    print("Loading backbone (quantized)...")
    quant_transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )

    # Apply LoRA to both
    if args.lora_dir:
        lora_kwargs = dict(
            r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
        )
        for tfm in [fp_transformer, quant_transformer]:
            tfm.add_adapter(LoraConfig(**lora_kwargs))
            tfm.enable_adapters()
            lora_sd = StableDiffusion3Pipeline.lora_state_dict(
                args.lora_dir, weight_name="transformer.safetensors", cache_dir=args.cache_dir)
            load_lora_state_dict(lora_sd, tfm)
            tfm = tfm.merge_and_unload()

        fp_transformer = fp_transformer.merge_and_unload()
        quant_transformer = quant_transformer.merge_and_unload()

    # Replace Linear -> QuantLinearW4A4 for both
    for tfm in [fp_transformer, quant_transformer]:
        replace_quant_layers(
            tfm, quant_scope=args.quant_scope, w_bits=4, a_bits=4,
            svdq_rank=args.rank, svdq_smooth_alpha=0.5,
            svdq_iterations=0,
            ffn_blocks=parse_ffn_blocks(args.quant_ffn_blocks),
        )

    # FP16 model: keep quant disabled (pass-through)
    fp_transformer = fp_transformer.to(device, dtype=weight_dtype).eval()
    set_quant_enabled(fp_transformer, False)

    # Quant model: apply cascade alpha, freeze, enable quant
    quant_transformer = quant_transformer.to(device, dtype=weight_dtype).eval()
    quant_transformer = build_quant_model(quant_transformer, cascade_report, device, weight_dtype)

    vae = vae.to(device, dtype=weight_dtype).eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"),
        map_location=device).to(dtype=weight_dtype)

    calib_names = get_image_names(args.calib_input_dir)[:args.calib_images]
    print(f"Calibration images: {len(calib_names)}")

    # Collect FP16 activation stats
    print("\n[1/2] Collecting FP16 activation stats...")
    fp_stats = collect_act_stats(fp_transformer, vae, calib_names,
                                 pooled_prompt_embeds, timesteps, weight_dtype,
                                 device, args.upscale, args.process_size, args.calib_images)

    # Collect post-quant activation stats
    print("\n[2/2] Collecting post-quant activation stats...")
    quant_stats = collect_act_stats(quant_transformer, vae, calib_names,
                                    pooled_prompt_embeds, timesteps, weight_dtype,
                                    device, args.upscale, args.process_size, args.calib_images)

    # Compare
    print(f"\n{'='*100}")
    print(f"Activation distribution shift: FP16 vs Post-Quant")
    print(f"{'Layer':<55} {'FP16 max':>10} {'Quant max':>10} {'ratio(Q/F)':>12} {'cosine_sim':>10} {'alpha_grid':>10} {'alpha_casc':>10}")
    print("-" * 130)

    quant_meta = {m["name"]: m for m in cascade_report["quant_meta"]}
    for name in sorted(fp_stats.keys()):
        if name not in quant_stats:
            continue

        fp_abs = fp_stats[name].float()
        q_abs = quant_stats[name].float()

        fp_max = fp_abs.max().item()
        q_max = q_abs.max().item()
        ratio = q_max / fp_max if fp_max > 0 else 0

        # Cosine similarity between channel distributions
        cos = torch.nn.functional.cosine_similarity(
            fp_abs.unsqueeze(0), q_abs.unsqueeze(0)).item()

        ga = quant_meta.get(name, {}).get("weight", {}).get("smooth_alpha", None)

        short = name if len(name) < 55 else "..." + name[-52:]

        # Highlight large shifts
        marker = ""
        if ratio > 1.10:
            marker = " <-- quant activation significantly MORE spread"
        elif ratio < 0.90:
            marker = " <-- quant activation MORE compressed"
        if cos < 0.95:
            marker += " (channel distribution shifted)"

        print(f"{short:<55} {fp_max:10.4f} {q_max:10.4f} {ratio:12.4f} {cos:10.4f} {ga}" if ga else f"{short:<55} {fp_max:10.4f} {q_max:10.4f} {ratio:12.4f} {cos:10.4f}")


if __name__ == "__main__":
    main()
