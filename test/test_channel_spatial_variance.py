#!/usr/bin/env python3
"""
Per-channel spatial variance analysis.

Tests the hypothesis: "spatial position information is encoded in a small
subset of feature channels; the remaining channels carry position-invariant
content."

Usage:
  python test/test_channel_spatial_variance.py --num_images 5
"""

import sys, os
_TINYSR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TINYSR_ROOT)

import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny


def _resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_TINYSR_ROOT, path))

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def list_images(root, max_num=50):
    import glob
    return sorted([f for ext in ("*.png","*.jpg","*.jpeg")
                    for f in glob.glob(os.path.join(root,"**",ext),recursive=True)])[:max_num]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def per_channel_spatial_std(X: torch.Tensor):
    """
    X: (N, D) — N tokens, D features
    Returns: (D,) array — spatial std of each channel across N positions,
             sorted descending, and the cumulative energy curve.
    """
    X = X.float()
    ch_std = X.std(dim=0)   # (D,) — per-channel spatial variability
    sorted_std, indices = ch_std.sort(descending=True)
    # cumulative fraction of total spatial variance
    cumsum = (sorted_std ** 2).cumsum(dim=0) / (sorted_std ** 2).sum()
    return ch_std, sorted_std, cumsum, indices


def run_analysis(model, vae, image_paths, pool_embeds, timestep, num_images, device, weight_dtype):
    """
    Run images through the flat model, collect per-block analysis.
    Returns: list of per-block stats aggregated across images.
    """
    model.eval()
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    # First pass to init
    dummy = torch.randn(1, 16, 64, 64, device=device, dtype=weight_dtype)
    _ = model(hidden_states=dummy, timestep=timestep, pooled_projections=pool_embeds, return_dict=False)

    n_blks = len(model.transformer_blocks)
    all_channel_stds = defaultdict(list)  # blk_idx → list of (1536,) arrays
    all_cumsums = defaultdict(list)        # blk_idx → list of (1536,) cumsum curves
    n_ok = 0

    for img_path in image_paths[:num_images]:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512), Image.BICUBIC)
            pixel = tensor_transform(img).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
            with torch.no_grad():
                latent = vae.encode(pixel).latents * vae.config.scaling_factor
                latent = F.interpolate(latent, size=(64, 64), mode="bilinear", align_corners=False)

            h = model.pos_embed(latent)
            for blk_idx, blk in enumerate(model.transformer_blocks):
                try:
                    h = blk.forward_(hidden_states=h)
                except AttributeError:
                    h = blk(hidden_states=h, temb=torch.zeros_like(h[:,:1,:1]))
                ch_std, sorted_std, cumsum, _ = per_channel_spatial_std(h[0])
                all_channel_stds[blk_idx].append(ch_std.cpu().numpy())
                all_cumsums[blk_idx].append(cumsum.cpu().numpy())
            n_ok += 1
        except Exception as e:
            continue
        torch.cuda.empty_cache()

    if n_ok == 0:
        print("  WARNING: no images processed, using random latent fallback.")
        with torch.no_grad():
            latent = torch.randn(1, 16, 64, 64, device=device, dtype=weight_dtype)
            h = model.pos_embed(latent)
            for blk_idx, blk in enumerate(model.transformer_blocks):
                try:
                    h = blk.forward_(hidden_states=h)
                except AttributeError:
                    h = blk(hidden_states=h, temb=torch.zeros_like(h[:,:1,:1]))
                ch_std, sorted_std, cumsum, _ = per_channel_spatial_std(h[0])
                all_channel_stds[blk_idx].append(ch_std.cpu().numpy())
                all_cumsums[blk_idx].append(cumsum.cpu().numpy())
        n_ok = 1

    # Aggregate: per-block stats
    results = []
    for blk_idx in range(n_blks):
        stds = np.stack(all_channel_stds[blk_idx])   # (n_images, 1536)
        cumsums = np.stack(all_cumsums[blk_idx])     # (n_images, 1536)

        avg_std = stds.mean(axis=0)                  # (1536,)
        avg_std_sorted = np.sort(avg_std)[::-1]      # descending
        avg_cumsum = (avg_std_sorted ** 2).cumsum() / (avg_std_sorted ** 2).sum()

        # Key metrics: X% of channels for Y% of spatial variance
        pcts = []
        for target_pct in [0.50, 0.70, 0.80, 0.90, 0.95]:
            nch = int((avg_cumsum < target_pct).sum() + 1)
            pcts.append((target_pct, nch, nch / 1536))
        results.append({
            "blk_idx": blk_idx,
            "total_spatial_var": float(avg_cumsum[-1]),
            "ch_for_50pct": pcts[0],
            "ch_for_70pct": pcts[1],
            "ch_for_80pct": pcts[2],
            "ch_for_90pct": pcts[3],
            "ch_for_95pct": pcts[4],
            "avg_cumsum_curve": avg_cumsum[:40].tolist(),  # first 40 points
        })
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results, n_images):
    print()
    print("=" * 85)
    print(f"  PER-CHANNEL SPATIAL VARIANCE ANALYSIS ({n_images} DIV2K images)")
    print("=" * 85)
    print(f"  {'Blk':>4}  {'50% var':>8}   {'70% var':>8}   {'80% var':>8}   {'90% var':>8}   {'95% var':>8}   {'Diagnosis':>20}")
    print(f"  {'':>4}  {'ch_ratio':>8}   {'ch_ratio':>8}   {'ch_ratio':>8}   {'ch_ratio':>8}   {'ch_ratio':>8}")
    print("-" * 85)

    for r in results:
        b = r["blk_idx"]
        p50 = r["ch_for_50pct"]
        p70 = r["ch_for_70pct"]
        p80 = r["ch_for_80pct"]
        p90 = r["ch_for_90pct"]
        p95 = r["ch_for_95pct"]

        n50, r50 = int(p50[1]), p50[2]
        n70, r70 = int(p70[1]), p70[2]
        n80, r80 = int(p80[1]), p80[2]
        n90, r90 = int(p90[1]), p90[2]
        n95, r95 = int(p95[1]), p95[2]

        # Diagnosis
        if r90 < 0.10:
            diag = "SPATIAL=MASSIVE skew"
        elif r90 < 0.20:
            diag = "SPATIAL=high skew"
        elif r90 < 0.35:
            diag = "SPATIAL=moderate skew"
        elif r90 < 0.50:
            diag = "SPATIAL=mild skew"
        else:
            diag = "SPATIAL=uniform"

        print(f"  {b:>4}  {n50:>3} ({r50:>5.1%})  {n70:>3} ({r70:>5.1%})  {n80:>3} ({r80:>5.1%})  {n90:>3} ({r90:>5.1%})  {n95:>3} ({r95:>5.1%})  {diag:>20}")

    print("-" * 85)
    avg90 = np.mean([r["ch_for_90pct"][2] for r in results])
    print(f"  Average: {avg90*1536:.0f} / 1536 channels ({avg90:.1%}) carry 90% of spatial variance")
    print()
    print("  INTERPRETATION:")
    print("  - Small ratio → few channels carry most spatial information")
    print("    → 'position' is encoded in a narrow feature subspace")
    print("    → Feasible to split: content on majority channels, position on minority")
    print("  - Large ratio → spatial info is spread across many channels → no clean split")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    p.add_argument("--vae_ckpt", type=str, default="checkpoint/vae/separable")
    p.add_argument("--embedding_dir", type=str, default="dataset/default")
    p.add_argument("--image_dir", type=str, default="dataset/DIV2K_train_LR_x8")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--json_out", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    weight_dtype = torch.float16

    print(f"[1/3] Loading models ...")
    model = TinySD3Transformer2DModel.from_pretrained(
        _resolve(args.ckpt), subfolder="transformer",
        torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, local_files_only=True).to(device)
    vae = AutoencoderTiny.from_pretrained(_resolve(args.vae_ckpt),
        torch_dtype=weight_dtype, local_files_only=True).to(device).eval()
    pool = torch.load(_resolve(os.path.join(args.embedding_dir, "pool_embeds.pt")),
                      map_location=device, weights_only=True)[:1].to(device, dtype=weight_dtype)
    t = torch.tensor([1000.], device=device, dtype=weight_dtype)

    print(f"[2/3] Running analysis on {args.num_images} images ...")
    images = list_images(_resolve(args.image_dir))
    results = run_analysis(model, vae, images, pool, t, args.num_images, device, weight_dtype)

    print_report(results, min(args.num_images, len(results[0]["avg_cumsum_curve"])))

    if args.json_out:
        out = [{k: (v.tolist() if hasattr(v,"tolist") else v) for k,v in r.items()} for r in results]
        with open(_resolve(args.json_out), "w") as f:
            json.dump(out, f, indent=2)
        print(f"  JSON → {args.json_out}")


if __name__ == "__main__":
    main()
