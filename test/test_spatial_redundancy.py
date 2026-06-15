#!/usr/bin/env python3
"""
Spatial redundancy analysis for TinySR pyramid stage planning.

Measures, per transformer block, how many tokens are needed to represent
the spatial information.  Answers: "at which block should I downsample,
and to what resolution?"

Metrics:
  1. Token effective rank: SVD of N×N Gram matrix → minimum viable N
  2. Token pairwise similarity: cosine similarity between neighbouring tokens
  3. Token variance per spatial position (identifies dead / uniform regions)

Usage:
  python test/test_spatial_redundancy.py \
    --ckpt checkpoint/tinybackbone/prune-12-merge-tinysr \
    --num_images 5
"""

import sys
import os

_TINYSR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TINYSR_ROOT)

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_TINYSR_ROOT, path))


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
    return sorted(imgs)[:max_num]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_spatial_metrics(hidden_states: torch.Tensor, grid_hw: int):
    """
    hidden_states: (B, N, D) where N = grid_hw × grid_hw, B = 1
    Returns a dict of per-block spatial metrics.
    """
    B, N, D = hidden_states.shape
    hw = int(N ** 0.5)
    if hw * hw != N:
        hw = grid_hw  # fallback to the expected grid

    # 1. Token effective rank via Gram SVD
    X = hidden_states[0].float()                             # (N, D)
    X_mean = X.mean(dim=0, keepdim=True)
    X_c = X - X_mean                                          # center
    gram = X_c @ X_c.T                                        # (N, N)
    S = torch.linalg.svdvals(gram)                            # (N,)
    total = (S ** 2).sum()
    if total < 1e-30:
        eff_rank_95 = 1
        eff_rank_99 = 1
    else:
        cumsum = (S ** 2).cumsum(dim=0) / total
        eff_rank_95 = int((cumsum < 0.95).sum().item() + 1)
        eff_rank_99 = int((cumsum < 0.99).sum().item() + 1)

    # also compute raw (uncentered) effective rank
    gram_raw = X @ X.T
    S_raw = torch.linalg.svdvals(gram_raw)
    total_raw = (S_raw ** 2).sum()
    if total_raw < 1e-30:
        eff_rank_raw = 1
    else:
        cumsum_raw = (S_raw ** 2).cumsum(dim=0) / total_raw
        eff_rank_raw = int((cumsum_raw < 0.95).sum().item() + 1)

    min_grid_95 = max(1, int(np.ceil(eff_rank_95 ** 0.5)))
    min_grid_99 = max(1, int(np.ceil(eff_rank_99 ** 0.5)))

    # 2. Token pairwise cosine similarity (neighbouring tokens)
    reshaped = X.reshape(hw, hw, D)
    # horizontal neighbours
    h_sim = F.cosine_similarity(
        reshaped[:-1, :, :].reshape(-1, D),
        reshaped[1:, :, :].reshape(-1, D), dim=-1)
    # vertical neighbours
    v_sim = F.cosine_similarity(
        reshaped[:, :-1, :].reshape(-1, D),
        reshaped[:, 1:, :].reshape(-1, D), dim=-1)
    mean_sim = float((h_sim.mean() + v_sim.mean()) / 2)
    min_sim = float(min(h_sim.min(), v_sim.min()))
    std_sim = float(min(h_sim.std(), v_sim.std()))  # per-image token std

    # 3. Token variance heat-map

    # 3. Token variance heat-map
    token_var = X.var(dim=-1)                                # (N,)
    token_var_map = token_var.reshape(hw, hw)
    # CV of spatial variance: if uniform → redundant spatial structure
    spatial_cv = float(token_var_map.std() / max(token_var_map.mean(), 1e-15))

    return {
        "grid_hw": hw,
        "num_tokens": N,
        "eff_rank_95": eff_rank_95,
        "eff_rank_99": eff_rank_99,
        "eff_rank_raw": eff_rank_raw,
        "min_grid_95": min_grid_95,
        "min_grid_99": min_grid_99,
        "rank_ratio": round(eff_rank_95 / N, 4),
        "rank_ratio_raw": round(eff_rank_raw / N, 4),
        "neighbor_sim_mean": round(mean_sim, 4),
        "neighbor_sim_min": round(min_sim, 4),
        "neighbor_sim_std": round(std_sim, 4),
        "spatial_cv": round(spatial_cv, 4),
    }


def collect_block_outputs_flat(model, latent, timestep, pool_embeds):
    """
    Run the flat model in eval mode.
    First pass initialises (deletes norm1, time_text_embed, temb, norm_out).
    Then manually iterate blocks via forward_() to collect outputs.
    """
    model.eval()
    with torch.no_grad():
        # First pass: initialise (deletes norm1 etc.)
        _ = model(hidden_states=latent.clone(), timestep=timestep,
                   pooled_projections=pool_embeds, return_dict=False)

        # Now the model is in steady-state: blocks use forward_ path.
        # Manually re-encode and iterate block-by-block to collect outputs.
        h = model.pos_embed(latent.clone())
        block_outputs = []
        for block in model.transformer_blocks:
            try:
                h = block.forward_(hidden_states=h)
            except AttributeError:
                # fallback (should not happen)
                h = block(hidden_states=h, temb=torch.zeros_like(h[:, :1, :1]))
            block_outputs.append(h.clone())
    return block_outputs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

COLORS = {
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "RESET": "\033[0m",
}


def _grid_recommend(rank_ratio, current_hw, neighbor_sim):
    """Heuristic: map effective rank ratio to recommended grid size."""
    if rank_ratio <= 0.10:
        factor = 8
    elif rank_ratio <= 0.20:
        factor = 4
    elif rank_ratio <= 0.35:
        factor = 2
    else:
        factor = 1
    rec_hw = max(1, current_hw // factor)
    if neighbor_sim > 0.95:
        rec_hw = max(1, current_hw // max(factor, 4))
    return rec_hw, factor


def format_report(all_block_metrics):
    """all_block_metrics: list of dicts, one per block, with aggregated per-image stats."""
    lines = []
    lines.append("=" * 105)
    lines.append("  SPATIAL REDUNDANCY REPORT  (token content compressibility)")
    lines.append("=" * 105)
    header = (f"{'Blk':>3} {'NbrSim':>8} {'NbrSim':>8} {'effRk':>6} {'effRk':>6} "
              f"{'rawRk':>6} {'minGrid':>8} {'Analysis':>25}")
    sub_hdr = (f"{'':>3} {'mean':>8} {'min':>8} {'(ctr)':>6} {'(raw)':>6} "
               f"{'(raw)':>6} {'(95%)':>8} {'':>25}")
    lines.append(header)
    lines.append(sub_hdr)
    lines.append("-" * 105)

    for blk_idx, m in enumerate(all_block_metrics):
        nsim_mean = m.get("neighbor_sim_mean", 0)
        nsim_min = m.get("neighbor_sim_min", 0)
        er_ctr = m.get("eff_rank_95", 0)
        er_raw = m.get("eff_rank_raw", 0)
        mg95 = m.get("min_grid_95", 0)

        # heuristic recommendation
        if nsim_mean > 0.97:
            level = "very safe to downsample 8×"
        elif nsim_mean > 0.93:
            level = "safe to downsample 4×"
        elif nsim_mean > 0.85:
            level = "safe to downsample 2×"
        else:
            level = "downsample risky"

        lines.append(
            f" {blk_idx:>3} {nsim_mean:>8.4f} {nsim_min:>8.4f} "
            f"{er_ctr:>6} {er_raw:>6} {er_raw:>6} {mg95:>4}×{mg95:<4} {level:>25}"
        )

    lines.append("-" * 105)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Spatial redundancy analysis for TinySR")
    p.add_argument("--ckpt", type=str, default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    p.add_argument("--vae_ckpt", type=str, default="checkpoint/vae/separable")
    p.add_argument("--embedding_dir", type=str, default="dataset/default")
    p.add_argument("--image_dir", type=str, default="dataset/test_image")
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--json_out", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    # ---- Load flat model (no LoRA, base weights) ----
    ckpt_path = _resolve(args.ckpt)
    print(f"[1/3] Loading flat teacher from {ckpt_path} ...")
    model = TinySD3Transformer2DModel.from_pretrained(
        ckpt_path, subfolder="transformer",
        torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, local_files_only=True,
    )
    model = model.to(device, dtype=weight_dtype).train()  # train() so forward not forward_
    n_blks = len(model.transformer_blocks)
    print(f"  Loaded. dim={model.inner_dim}, blocks={n_blks}")

    # ---- Load VAE + embeddings ----
    print(f"[2/3] Loading VAE + embeddings ...")
    vae = None
    vae_ckpt_path = _resolve(args.vae_ckpt)
    if os.path.exists(vae_ckpt_path):
        try:
            vae = AutoencoderTiny.from_pretrained(vae_ckpt_path, torch_dtype=weight_dtype, local_files_only=True)
            vae = vae.to(device, dtype=weight_dtype).eval()
        except Exception as e:
            print(f"  VAE load failed ({e}), using random latents.")

    pool_embeds_path = _resolve(os.path.join(args.embedding_dir, "pool_embeds.pt"))
    if os.path.exists(pool_embeds_path):
        pool_embeds = torch.load(pool_embeds_path, map_location=device, weights_only=True)
        if pool_embeds.dim() == 2:
            pool_embeds = pool_embeds[:1]
    else:
        pool_embeds = torch.zeros(1, 2048, device=device, dtype=weight_dtype)
    pool_embeds = pool_embeds.to(device, dtype=weight_dtype)
    timestep = torch.tensor([1000.0], device=device, dtype=weight_dtype)

    # ---- Collect ----
    print(f"[3/3] Running {args.num_images} forward passes and collecting per-block features ...")
    images = list_images(_resolve(args.image_dir), max_num=args.num_images * 3)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    all_metrics = defaultdict(list)
    sample_count = 0

    for img_path in images:
        if sample_count >= args.num_images:
            break
        try:
            img = Image.open(img_path).convert("RGB")
            if vae is not None:
                img = img.resize((512, 512), Image.BICUBIC)
                pixel = tensor_transform(img).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
                with torch.no_grad():
                    latent = vae.encode(pixel).latents * vae.config.scaling_factor
                    H = W = 64
                    if latent.shape[-1] != H:
                        latent = F.interpolate(latent, size=(H, W), mode="bilinear", align_corners=False)
            else:
                img = img.resize((128, 128), Image.BICUBIC)
                pixel = tensor_transform(img).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
                latent = torch.randn(1, 16, 64, 64, device=device, dtype=weight_dtype)
        except Exception as e:
            continue

        block_outs = collect_block_outputs_flat(model, latent, timestep, pool_embeds)
        for blk_idx, h in enumerate(block_outs):
            metrics = compute_spatial_metrics(h, grid_hw=32)  # 64/2=32 after patch
            for k, v in metrics.items():
                all_metrics[k].append(v)
        sample_count += 1
        print(f"  [{sample_count}/{args.num_images}] {os.path.basename(img_path)}")

    if sample_count == 0:
        print("  No images processed, using random latent.")
        latent = torch.randn(1, 16, 64, 64, device=device, dtype=weight_dtype)
        block_outs = collect_block_outputs_flat(model, latent, timestep, pool_embeds)
        for blk_idx, h in enumerate(block_outs):
            metrics = compute_spatial_metrics(h, grid_hw=32)
            for k, v in metrics.items():
                all_metrics[k].append(v)

    # ---- Aggregate across images: per-block statistics ----
    avg_metrics = []
    for blk_idx in range(n_blks):
        blk_data = defaultdict(list)
        for metric_name, vals in all_metrics.items():
            for img_idx in range(sample_count):
                idx = img_idx * n_blks + blk_idx
                if idx < len(vals):
                    blk_data[metric_name].append(vals[idx])
        blk = {}
        for k, vlist in blk_data.items():
            if not vlist:
                continue
            arr = np.array(vlist, dtype=np.float64)
            if k in ("grid_hw", "num_tokens", "eff_rank_95", "eff_rank_99",
                     "eff_rank_raw", "min_grid_95", "min_grid_99"):
                blk[k] = int(np.median(arr))
            else:
                blk[k] = round(float(np.mean(arr)), 4)
        avg_metrics.append(blk)

    # ---- Report ----
    print(format_report(avg_metrics))

    # ---- JSON ----
    if args.json_out:
        json_path = _resolve(args.json_out)
        with open(json_path, "w") as f:
            json.dump(avg_metrics, f, indent=2, ensure_ascii=False)
        print(f"  JSON saved to {json_path}")


if __name__ == "__main__":
    main()
