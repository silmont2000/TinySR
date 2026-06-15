#!/usr/bin/env python3
"""
Linear layer redundancy analysis for TinySR.

Two analyses:
  1. SVD effective rank — how many singular values capture 95%/99% of the weight matrix energy.
     High redundancy → effective rank ≪ full rank → mult can be reduced.
  2. Per-channel activation utilization — for FFN intermediate layers (e.g. 6144-dim),
     how many channels actually carry signal vs. sitting near zero.

Usage:
  python test/test_linear_redundancy.py \
    --ckpt checkpoint/tinybackbone/prune-12-merge-tinysr \
    --num_images 5
"""

import sys
import os

# Resolve TinySR root so relative checkpoint paths work regardless of cwd
_TINYSR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TINYSR_ROOT)

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from torchvision import transforms
from PIL import Image


def _resolve(path: str) -> str:
    """Resolve a path: if absolute, keep it; otherwise make it relative to TinySR root."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_TINYSR_ROOT, path))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root: str, max_num: int = 20) -> list:
    import glob
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        imgs.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    imgs = sorted(imgs)
    return imgs[:max_num] if max_num > 0 else imgs


# ---------------------------------------------------------------------------
# Part 1: SVD effective rank
# ---------------------------------------------------------------------------

def svd_effective_rank(W: torch.Tensor, thresholds=(0.90, 0.95, 0.99)):
    """
    Compute effective rank of weight matrix at given energy thresholds.
    W: (out_features, in_features)
    Returns dict with effective ranks at each threshold, ratio, spectrum decay.
    """
    W = W.float()
    with torch.no_grad():
        # torch.linalg.svdvals is memory-efficient (only returns S, not U/V)
        S = torch.linalg.svdvals(W)
    total = (S ** 2).sum()
    if total < 1e-30:
        return {f"erank_{int(t*100)}": 0 for t in thresholds} | {
            "full_rank": 0, "ratio_95": 0.0, "top5_energy": 0.0, "decay": 0.0
        }
    cumsum = (S ** 2).cumsum(dim=0) / total
    result = {"full_rank": int(S.shape[0])}
    for t in thresholds:
        erank = int((cumsum < t).sum().item() + 1)
        result[f"erank_{int(t*100)}"] = erank
    result["ratio_95"] = round(result["erank_95"] / max(1, result["full_rank"]), 4)
    result["top5_energy"] = round(cumsum[min(4, cumsum.shape[0] - 1)].item(), 4)
    result["decay"] = round((S[0] / max(S[-1], 1e-15)).item(), 1) if S.shape[0] > 1 else 0.0
    return result


def run_svd_analysis(model):
    """
    Run SVD on all nn.Linear weight matrices, classify into categories.
    Returns (results_dict, summary_stats).
    """
    results = {}
    categories = {"ffn_up": [], "ffn_down": [], "q": [], "k": [], "v": [],
                  "out": [], "norm": [], "embed": [], "proj_out": [], "other": []}

    def _classify(name):
        if "ff" in name and "net.0" in name and "proj" in name:
            return "ffn_up"
        if "ff" in name and "net.2" in name:
            return "ffn_down"
        if "to_q" in name:
            return "q"
        if "to_k" in name:
            return "k"
        if "to_v" in name:
            return "v"
        if "to_out.0" in name:
            return "out"
        if "norm" in name.lower() and "linear" in name:
            return "norm"
        if "embed" in name.lower() or "time_text" in name.lower():
            return "embed"
        if "proj_out" in name and "pos" not in name:
            return "proj_out"
        return "other"

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight.data
        info = svd_effective_rank(W)
        info["shape"] = list(W.shape)
        info["params"] = int(W.numel())
        cat = _classify(name)
        info["category"] = cat
        categories[cat].append(info["ratio_95"])
        results[name] = info

    summary = {}
    for cat, ratios in categories.items():
        if ratios:
            summary[f"{cat}_count"] = len(ratios)
            summary[f"{cat}_avg_ratio_95"] = round(np.mean(ratios), 4)
            summary[f"{cat}_min_ratio_95"] = round(np.min(ratios), 4)
            summary[f"{cat}_max_ratio_95"] = round(np.max(ratios), 4)

    # overall FFN redundancy
    ffn_ratios = categories["ffn_up"] + categories["ffn_down"]
    if ffn_ratios:
        summary["ffn_overall_avg_ratio_95"] = round(np.mean(ffn_ratios), 4)
        # heuristic: safe mult = current_mult * avg_ratio (crude but directional)
        summary["ffn_suggested_mult_factor"] = round(np.mean(ffn_ratios), 2)

    return results, summary, categories


# ---------------------------------------------------------------------------
# Part 2: Per-channel activation analysis
# ---------------------------------------------------------------------------

class ChannelActivationCollector:
    """Hook-based collector that stores per-channel mean absolute activations."""

    def __init__(self):
        # layer_name -> list of (C,) arrays (per-channel mean abs)
        self.channel_means = defaultdict(list)
        self.handles = []

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            x = output if output is not None else (input[0] if input else None)
            if x is None or not isinstance(x, torch.Tensor):
                return
            x = x.detach().float()
            if x.dim() >= 2:
                *_, C = x.shape
                ch_abs = x.abs().reshape(-1, C).mean(dim=0)  # (C,)
                self.channel_means[layer_name].append(ch_abs.cpu().numpy())
        return hook

    def register(self, model, target_substrings=("ff.net.0", "ff.net.2")):
        """Register forward hooks on modules whose name contains any target substring."""
        for name, module in model.named_modules():
            if any(sub in name for sub in target_substrings):
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def compute_all(self):
        """Compute per-layer stats from collected channel means."""
        stats = {}
        for name, samples in sorted(self.channel_means.items()):
            if not samples:
                continue
            stacked = np.stack(samples, axis=0)          # (n_samples, C)
            ch_avg = stacked.mean(axis=0)                 # (C,)
            ch_std = stacked.std(axis=0)
            C = len(ch_avg)

            total_energy = ch_avg.sum()
            if total_energy < 1e-30:
                stats[name] = {"channels": C, "dead_ratio": 1.0, "active_90": 0,
                               "concentration": 0.0, "avg_magnitude": 0.0}
                continue

            # dead channels: magnitude < 1% of mean magnitude
            threshold = ch_avg.mean() * 0.01
            dead_ratio = float((ch_avg < threshold).mean())

            # how many channels capture 90% of total activation energy
            sorted_avg = np.sort(ch_avg)[::-1]
            cumsum = sorted_avg.cumsum() / sorted_avg.sum()
            active_90 = int((cumsum < 0.9).sum() + 1)

            # coefficient of variation across channels (how uneven is utilization)
            cv = float(ch_std.mean() / max(ch_avg.mean(), 1e-15))

            stats[name] = {
                "channels": C,
                "dead_ratio": round(dead_ratio, 4),
                "active_90": active_90,
                "concentration": round(active_90 / C, 4),
                "avg_magnitude": round(float(ch_avg.mean()), 6),
                "cv": round(cv, 4),
            }
        return stats


def run_activation_analysis(model, image_paths, device, weight_dtype, pool_embeds, timestep, num_images=10):
    """
    Run activation collection on the flat model.
    First forward pass initializes (deletes norm1, etc.) — we collect on subsequent passes.
    """
    collector = ChannelActivationCollector()
    collector.register(model, target_substrings=("ff.net.0", "ff.net.2"))

    tensor_transform = transforms.Compose([transforms.ToTensor()])

    # First pass: initializes the model (deletes norm1, time_text_embed, norm_out)
    # Use a dummy latent — same shape as actual VAE output
    H = W = 64  # sample_size
    dummy = torch.randn(1, 16, H, W, device=device, dtype=weight_dtype)
    _ = model(hidden_states=dummy, timestep=timestep, pooled_projections=pool_embeds, return_dict=False)

    # Now collect on subsequent passes
    collector.channel_means.clear()  # discard first pass data (may be noisy due to init)

    for i, img_path in enumerate(image_paths[:num_images]):
        try:
            img = Image.open(img_path).convert("RGB")
            # resize to 128×128 (input size, VAE will encode to 64×64 latent at 16ch)
            img = img.resize((128, 128), Image.BICUBIC)
            pixel = tensor_transform(img).unsqueeze(0).to(device, dtype=weight_dtype)
            pixel = pixel * 2 - 1
            # do a simple forward (skip VAE encode to keep it simple):
            # use model directly with random latent of correct shape
            # For activation analysis, the relative channel utilization mostly depends
            # on weight structure, not on exact input distribution.
            # But using actual VAE latents is more accurate.
            latent = torch.randn(1, 16, H, W, device=device, dtype=weight_dtype)
            with torch.no_grad():
                _ = model(hidden_states=latent, timestep=timestep,
                          pooled_projections=pool_embeds, return_dict=False)
        except Exception as e:
            print(f"  skip {os.path.basename(img_path)}: {e}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    collector.remove()
    return collector.compute_all()


def run_activation_analysis_with_vae(model, vae, image_paths, device, weight_dtype,
                                      pool_embeds, timestep, num_images=10):
    """Same as above but encodes images through VAE for realistic latents."""
    collector = ChannelActivationCollector()
    collector.register(model, target_substrings=("ff.net.0", "ff.net.2"))

    tensor_transform = transforms.Compose([transforms.ToTensor()])

    H = W = 64
    # First forward pass to initialize
    dummy = torch.randn(1, 16, H, W, device=device, dtype=weight_dtype)
    _ = model(hidden_states=dummy, timestep=timestep, pooled_projections=pool_embeds, return_dict=False)
    collector.channel_means.clear()

    for i, img_path in enumerate(image_paths[:num_images]):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512), Image.BICUBIC)  # VAE expects 512
            pixel = tensor_transform(img).unsqueeze(0).to(device, dtype=weight_dtype)
            pixel = pixel * 2 - 1
            with torch.no_grad():
                latent = vae.encode(pixel).latents * vae.config.scaling_factor
                if latent.shape[-1] != H or latent.shape[-2] != W:
                    latent = torch.nn.functional.interpolate(
                        latent, size=(H, W), mode="bilinear", align_corners=False)
                _ = model(hidden_states=latent, timestep=timestep,
                          pooled_projections=pool_embeds, return_dict=False)
        except Exception as e:
            print(f"  skip {os.path.basename(img_path)}: {e}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    collector.remove()
    return collector.compute_all()


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_svd_table(results, title="SVD EFFECTIVE RANK ANALYSIS"):
    lines = ["=" * 110,
             f"  {title} (energy threshold = 95%)",
             "=" * 110,
             f"{'Layer':<58} {'Shape':<18} {'Full':>5} {'Eff95':>6} {'Ratio':>7} {'Top5':>7} {'Decay':>9}",
             "-" * 110]

    for name, info in sorted(results.items()):
        shape_str = str(info["shape"])
        lines.append(
            f"{name:<58} {shape_str:<18} {info['full_rank']:>5} "
            f"{info['erank_95']:>6} {info['ratio_95']:>7.3f} "
            f"{info['top5_energy']:>7.3f} {info['decay']:>9.1f}"
        )
    lines.append("-" * 110)
    return "\n".join(lines)


def format_ffn_svd_summary(results, categories):
    """Show per-block FFN effective rank for quick comparison."""
    lines = ["", "FFN PER-BLOCK SVD SUMMARY:", "-" * 60,
             f"{'Block':<35} {'Up Ratio':>10} {'Down Ratio':>12}",
             "-" * 60]

    # extract block indices and group
    ffn_up = {k: v for k, v in results.items() if v["category"] == "ffn_up"}
    ffn_down = {k: v for k, v in results.items() if v["category"] == "ffn_down"}

    for i in range(12):
        up_key = f"transformer_blocks.{i}.ff.net.0.proj"
        down_key = f"transformer_blocks.{i}.ff.net.2"
        up_r = ffn_up.get(up_key, {}).get("ratio_95", "N/A")
        down_r = ffn_down.get(down_key, {}).get("ratio_95", "N/A")
        up_str = f"{up_r:.3f}" if isinstance(up_r, float) else str(up_r)
        down_str = f"{down_r:.3f}" if isinstance(down_r, float) else str(down_r)
        lines.append(f"  block {i:<31} {up_str:>10} {down_str:>12}")

    avg_up = np.mean([v["ratio_95"] for v in ffn_up.values()]) if ffn_up else 0
    avg_down = np.mean([v["ratio_95"] for v in ffn_down.values()]) if ffn_down else 0
    lines.append("-" * 60)
    lines.append(f"  {'AVERAGE':<31} {avg_up:>10.3f} {avg_down:>12.3f}")
    lines.append(f"  Suggested mult factor (avg ratio * 4): {4 * (avg_up + avg_down) / 2:.1f}")
    return "\n".join(lines)


def format_activation_table(act_stats):
    lines = ["", "=" * 95,
             "  PER-CHANNEL ACTIVATION UTILIZATION (FFN layers)",
             "=" * 95,
             f"{'Layer':<55} {'Ch':>5} {'Dead%':>7} {'Act90':>6} {'Conc':>7} {'CV':>7}",
             "-" * 95]

    for name, info in act_stats.items():
        lines.append(
            f"{name:<55} {info['channels']:>5} {info['dead_ratio']*100:>6.1f}% "
            f"{info['active_90']:>6} {info['concentration']:>7.3f} {info['cv']:>7.3f}"
        )
    lines.append("-" * 95)

    # summary
    ffn0_stats = [v for k, v in act_stats.items() if "net.0" in k]
    ffn2_stats = [v for k, v in act_stats.items() if "net.2" in k]
    if ffn0_stats:
        avg_dead = np.mean([s["dead_ratio"] for s in ffn0_stats])
        avg_conc = np.mean([s["concentration"] for s in ffn0_stats])
        avg_active = np.mean([s["active_90"] for s in ffn0_stats])
        avg_ch = np.mean([s["channels"] for s in ffn0_stats])
        lines.append(f"  FFN up avg: {int(avg_ch)} ch, {avg_dead*100:.1f}% dead, "
                     f"active_90={avg_active:.0f} ({avg_conc:.3f} of total)")
        implied_mult = avg_conc * 4
        lines.append(f"  Implied effective mult: {implied_mult:.1f} (current=4.0)")
    if ffn2_stats:
        avg_conc2 = np.mean([s["concentration"] for s in ffn2_stats])
        avg_active2 = np.mean([s["active_90"] for s in ffn2_stats])
        avg_ch2 = np.mean([s["channels"] for s in ffn2_stats])
        lines.append(f"  FFN down avg: {int(avg_ch2)} ch, active_90={avg_active2:.0f} ({avg_conc2:.3f} of total)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Linear redundancy analysis for TinySR")
    p.add_argument("--ckpt", type=str, default="checkpoint/tinybackbone/prune-12-merge-tinysr",
                   help="Path to flat TinySR checkpoint (contains transformer/ subdir)")
    p.add_argument("--vae_ckpt", type=str, default="checkpoint/vae/separable",
                   help="Path to VAE checkpoint")
    p.add_argument("--embedding_dir", type=str, default="dataset/default",
                   help="Path to pooled prompt embeddings")
    p.add_argument("--image_dir", type=str, default="dataset/test_image",
                   help="Directory with test images for activation collection")
    p.add_argument("--num_images", type=int, default=5,
                   help="Number of images for activation collection")
    p.add_argument("--no_activation", action="store_true",
                   help="Skip activation analysis (do SVD only)")
    p.add_argument("--svd_threshold", type=float, default=0.95,
                   help="Energy threshold for effective rank (default 0.95)")
    p.add_argument("--json_out", type=str, default="",
                   help="Optional path to save JSON report")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    # ---- Load model (flat backbone, no LoRA) ----
    ckpt_path = _resolve(args.ckpt)
    print(f"[1/3] Loading model from {ckpt_path} ...")
    model = TinySD3Transformer2DModel.from_pretrained(
        ckpt_path, subfolder="transformer",
        torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, local_files_only=True,
    )
    model = model.to(device, dtype=weight_dtype).eval()
    param_cnt = sum(p.numel() for p in model.parameters())
    print(f"  Loaded.  Parameters: {param_cnt/1e6:.1f}M,  dim: {model.inner_dim},  blocks: {len(model.transformer_blocks)}")

    # ---- Part 1: SVD analysis ----
    print("\n[2/3] Running SVD effective rank analysis ...")
    svd_results, svd_summary, categories = run_svd_analysis(model)

    print(format_svd_table(svd_results))
    print(format_ffn_svd_summary(svd_results, categories))

    # category summary
    print("\nSVD SUMMARY BY CATEGORY:")
    for k, v in sorted(svd_summary.items()):
        print(f"  {k}: {v}")

    # ---- Part 2: Activation analysis ----
    act_stats = {}
    if not args.no_activation:
        print(f"\n[3/3] Running per-channel activation analysis ({args.num_images} images) ...")

        pool_embeds_path = _resolve(os.path.join(args.embedding_dir, "pool_embeds.pt"))
        if os.path.exists(pool_embeds_path):
            pool_embeds = torch.load(pool_embeds_path, map_location=device, weights_only=True)
            if pool_embeds.dim() == 2:
                pool_embeds = pool_embeds[:1]
        else:
            print(f"  WARNING: {pool_embeds_path} not found, using zeros.")
            pool_embeds = torch.zeros(1, 2048, device=device, dtype=weight_dtype)
        pool_embeds = pool_embeds.to(device, dtype=weight_dtype)
        timestep = torch.tensor([1000.0], device=device, dtype=weight_dtype)

        # Try VAE path; fall back to random latent if VAE unavailable
        vae = None
        use_vae = False
        vae_ckpt_path = _resolve(args.vae_ckpt)
        if os.path.exists(vae_ckpt_path):
            try:
                vae = AutoencoderTiny.from_pretrained(vae_ckpt_path, torch_dtype=weight_dtype, local_files_only=True)
                vae = vae.to(device, dtype=weight_dtype).eval()
                use_vae = True
                print("  VAE loaded for realistic latent collection.")
            except Exception as e:
                print(f"  VAE load failed ({e}), falling back to random latents.")

        images = list_images(_resolve(args.image_dir), max_num=args.num_images * 3)  # oversample
        if not images:
            print("  No images found, using random latents.")

        if use_vae and images:
            act_stats = run_activation_analysis_with_vae(
                model, vae, images, device, weight_dtype,
                pool_embeds, timestep, args.num_images)
        else:
            act_stats = run_activation_analysis(
                model, images or ["dummy"], device, weight_dtype,
                pool_embeds, timestep, max(args.num_images, 1))

        print(format_activation_table(act_stats))

    # ---- JSON report (optional) ----
    if args.json_out:
        report = {
            "config": vars(args),
            "svd_results": {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                                 for kk, vv in v.items()}
                            for k, v in svd_results.items()},
            "svd_summary": svd_summary,
            "activation_stats": act_stats,
        }
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved to {args.json_out}")

    # ---- Final recommendations ----
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)

    ffn_up_ratios = categories.get("ffn_up", [])
    ffn_down_ratios = categories.get("ffn_down", [])
    avg_ffn_ratio = np.mean(ffn_up_ratios + ffn_down_ratios) if (ffn_up_ratios or ffn_down_ratios) else 0
    suggested_mult = 4.0 * avg_ffn_ratio

    if avg_ffn_ratio < 0.5:
        print(f"  FFN effective rank ratio = {avg_ffn_ratio:.3f} → SEVERE redundancy.")
        print(f"  Safe to reduce mult from 4.0 to 2.0–2.5 immediately.")
    elif avg_ffn_ratio < 0.65:
        print(f"  FFN effective rank ratio = {avg_ffn_ratio:.3f} → MODERATE redundancy.")
        print(f"  mult 4.0 → 2.5 is safe. Try 2.0 with distillation monitoring.")
    elif avg_ffn_ratio < 0.80:
        print(f"  FFN effective rank ratio = {avg_ffn_ratio:.3f} → MILD redundancy.")
        print(f"  mult 4.0 → 3.0 is safe. 2.5 may need careful distillation.")
    else:
        print(f"  FFN effective rank ratio = {avg_ffn_ratio:.3f} → LOW redundancy.")
        print(f"  Keep mult at 4.0 or try 3.5. Large reductions risky.")

    if act_stats:
        ffn0_conc = [s["concentration"] for k, s in act_stats.items() if "net.0" in k]
        ffn0_dead = [s["dead_ratio"] for k, s in act_stats.items() if "net.0" in k]
        if ffn0_conc:
            avg_conc = np.mean(ffn0_conc)
            avg_dead = np.mean(ffn0_dead)
            implied = avg_conc * 4.0
            print(f"  Activation concentration = {avg_conc:.3f} (90% energy in {int(avg_conc*6144)} of 6144 ch)")
            print(f"  Dead channels = {avg_dead*100:.1f}%")
            print(f"  Cross-check: SVD suggests mult≈{suggested_mult:.1f}, activation suggests mult≈{implied:.1f}")

    print()


if __name__ == "__main__":
    main()
