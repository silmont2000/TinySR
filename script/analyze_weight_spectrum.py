#!/usr/bin/env python3
"""Spectral (SVD) analysis of TinySR Linear weight matrices.

Reads the layer list from a layer_means.pt dump (produced by
script/dump_layer_means.py) and loads the corresponding weight matrices W from
the merged_backbone safetensors. For every Linear weight it computes five
quantization-friendliness diagnostics:

  1. Singular value spectrum      — sigma_i (log-scale decay curve)
  2. Energy retention ratio       — E_k = sum_{i<=k} sigma_i^2 / sum sigma_i^2,
                                     plus k reaching 90% / 95% / 99%
  3. Effective rank (entropy)     — exp(-sum p_i log p_i),  p_i = sigma_i / sum sigma
  4. Stable rank                  — ||W||_F^2 / ||W||_2^2 = sum sigma_i^2 / sigma_1^2
  5. Condition number             — sigma_max / sigma_min

Outputs: a metrics CSV plus PNG figures (spectrum overlay, energy-retention
overlay, per-layer bar charts, and a block x layer-type heatmap).

Usage:
    python script/analyze_weight_spectrum.py \\
        --means outputs/layer_means.pt \\
        --weights "/path/to/merged_backbone/diffusion_pytorch_model.safetensors" \\
        --output_dir outputs/spectrum
"""

import argparse
import csv
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

DEFAULT_WEIGHTS = (
    "/Users/xieboyang/Documents/Tinysr/可以保留的数据/好/"
    "sweep_tier-2-3/sweep_tier-2-3/merged_backbone/"
    "diffusion_pytorch_model.safetensors"
)


def parse_args():
    p = argparse.ArgumentParser(description="SVD spectral analysis of Linear weights")
    p.add_argument("--means", type=str, default="outputs/layer_means.pt",
                   help="layer_means.pt (used only for the layer-name list)")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                   help="Path to merged_backbone safetensors")
    p.add_argument("--output_dir", "-o", type=str, default="outputs/spectrum")
    p.add_argument("--energy_thresholds", type=float, nargs="*",
                   default=[0.90, 0.95, 0.99])
    p.add_argument("--filter", type=str, default="ff.net",
                   help="Only analyze layers whose name contains this substring "
                        "(empty string = no filter)")
    return p.parse_args()


# ── metric computation ─────────────────────────────────────────

def analyze_weight(W: torch.Tensor, thresholds):
    """Return (sigma, energy_curve, metrics_dict) for a weight matrix."""
    W = W.float()
    sigma = torch.linalg.svdvals(W)          # descending, shape (min(out,in),)
    sigma = sigma.clamp_min(0)
    s = sigma.numpy()
    s2 = s ** 2
    total_e = s2.sum()

    energy = np.cumsum(s2) / max(total_e, 1e-30)   # E_k, k=1..n

    # thresholds -> smallest k reaching each level
    k_at = {}
    for t in thresholds:
        idx = int(np.searchsorted(energy, t) + 1)
        k_at[t] = min(idx, len(s))

    sigma_max = float(s[0]) if len(s) else 0.0
    sigma_min = float(s[-1]) if len(s) else 0.0
    cond = sigma_max / max(sigma_min, 1e-12)

    stable_rank = float(s2.sum() / max(s2[0], 1e-30))

    p = s / max(s.sum(), 1e-30)
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    metrics = {
        "n": len(s),
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "cond": cond,
        "stable_rank": stable_rank,
        "eff_rank": eff_rank,
    }
    for t in thresholds:
        metrics[f"k{int(t * 100)}"] = k_at[t]
    return s, energy, metrics


# ── heatmap grid helpers ───────────────────────────────────────

_BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.(.+)")


def block_and_sub(layer_name):
    m = _BLOCK_RE.match(layer_name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, layer_name


# ── plotting ───────────────────────────────────────────────────

def _color_for(block, n_blocks):
    if block is None:
        return "#888888"
    return plt.cm.viridis(block / max(n_blocks - 1, 1))


def plot_spectrum(results, n_blocks, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, (s, _e, _m, blk) in results.items():
        if len(s) < 2:
            continue
        y = s / s[0]
        x = np.arange(1, len(s) + 1) / len(s)
        ax.plot(x, y, color=_color_for(blk, n_blocks), alpha=0.35, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("normalized index  (i / n)")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$  (log)")
    ax.set_title("Singular value spectrum decay (all Linear weights)")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(0, n_blocks - 1))
    fig.colorbar(sm, ax=ax, label="block index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_energy(results, n_blocks, out_path, thresholds=(0.90, 0.95, 0.99)):
    """Small-multiples: one subplot per layer so curves stay distinguishable."""
    items = [(name, e, blk) for name, (_s, e, _m, blk) in results.items()
             if len(e) >= 2]
    if not items:
        return
    n = len(items)
    ncols = 4 if n > 6 else min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.4),
                             squeeze=False)
    for idx, (name, e, blk) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        k = np.arange(1, len(e) + 1)
        x = k / len(e)
        color = _color_for(blk, n_blocks)
        ax.plot(x, e, color=color, linewidth=1.4)
        for t in thresholds:
            kt = int(np.searchsorted(e, t) + 1)
            kt = min(kt, len(e))
            ax.axhline(t, color="grey", linestyle=":", linewidth=0.5)
            ax.axvline(kt / len(e), color="red", linestyle="--", linewidth=0.6)
            ax.annotate(f"k{int(t*100)}={kt}", (kt / len(e), t),
                        fontsize=5, color="red",
                        xytext=(2, -6), textcoords="offset points")
        ax.set_title(name.replace("transformer_blocks.", "b"), fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=6)
        if idx % ncols == 0:
            ax.set_ylabel(r"$E_k$", fontsize=7)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("k / n", fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Energy retention per layer", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_bars(rows, out_path):
    names = [r["layer"] for r in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(names) * 0.16), 10),
                             sharex=True)

    axes[0].bar(x, [r["cond"] for r in rows], color="#e03131")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("condition number (log)")
    axes[0].set_title("Per-layer quantization-friendliness "
                      "(lower cond / higher stable&eff rank = friendlier)")

    axes[1].bar(x, [r["stable_rank"] for r in rows], color="#1c7ed6")
    axes[1].set_ylabel("stable rank")

    axes[2].bar(x, [r["eff_rank"] for r in rows], color="#2f9e44")
    axes[2].set_ylabel("effective rank")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=90, fontsize=4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_heatmap(rows, metric_key, title, out_path):
    """Grid heatmap over blocks (rows) x repeating layer-type (cols)."""
    grid = {}
    subs = []
    blocks = set()
    for r in rows:
        blk, sub = r["_block"], r["_sub"]
        if blk is None:
            continue
        blocks.add(blk)
        if sub not in subs:
            subs.append(sub)
        grid[(blk, sub)] = r[metric_key]
    if not blocks:
        return
    blocks = sorted(blocks)
    subs = sorted(subs)
    mat = np.full((len(blocks), len(subs)), np.nan)
    for i, b in enumerate(blocks):
        for j, sub in enumerate(subs):
            if (b, sub) in grid:
                mat[i, j] = grid[(b, sub)]

    fig, ax = plt.subplots(figsize=(max(8, len(subs) * 1.1),
                                    max(5, len(blocks) * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"block {b}" for b in blocks], fontsize=8)
    ax.set_title(title)
    for i in range(len(blocks)):
        for j in range(len(subs)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}" if mat[i, j] > 10
                        else f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white")
    fig.colorbar(im, ax=ax, label=metric_key)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Layer list: from a layer_means.pt dump if present, otherwise enumerate
    # every "*.weight" key in the weights file itself.
    if args.means and os.path.exists(args.means):
        payload = torch.load(args.means, map_location="cpu")
        layer_names = payload["layers"]
        src = args.means
    else:
        with safe_open(args.weights, framework="pt") as f:
            layer_names = sorted(k[:-len(".weight")] for k in f.keys()
                                 if k.endswith(".weight"))
        src = args.weights
    if args.filter:
        layer_names = [n for n in layer_names if args.filter in n]
    print(f"[layers] {len(layer_names)} from {src}"
          + (f" (filter='{args.filter}')" if args.filter else ""))

    results = {}   # name -> (sigma, energy, metrics, block)
    rows = []
    n_blocks = 0

    with safe_open(args.weights, framework="pt") as f:
        available = set(f.keys())
        for name in layer_names:
            key = f"{name}.weight"
            if key not in available:
                print(f"  [WARN] no weight for {name}")
                continue
            W = f.get_tensor(key)
            if W.ndim != 2:
                print(f"  [WARN] {name} weight ndim={W.ndim}, skipped")
                continue
            s, energy, metrics = analyze_weight(W, args.energy_thresholds)
            blk, sub = block_and_sub(name)
            if blk is not None:
                n_blocks = max(n_blocks, blk + 1)
            results[name] = (s, energy, metrics, blk)

            row = {"layer": name,
                   "shape": f"{W.shape[0]}x{W.shape[1]}",
                   "_block": blk, "_sub": sub}
            row.update(metrics)
            rows.append(row)
            print(f"  {name:<48s} cond={metrics['cond']:.1f} "
                  f"stable={metrics['stable_rank']:.1f} "
                  f"eff={metrics['eff_rank']:.1f}")

    if not rows:
        raise SystemExit("No weights analyzed.")

    # ── CSV ────────────────────────────────────────────────────
    csv_keys = ["layer", "shape", "n", "sigma_max", "sigma_min", "cond",
                "stable_rank", "eff_rank"]
    csv_keys += [f"k{int(t * 100)}" for t in args.energy_thresholds]
    csv_path = os.path.join(args.output_dir, "weight_spectrum.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {len(rows)} layers -> {csv_path}")

    # ── raw singular values (for further analysis) ─────────────
    sv_dump = {name: torch.from_numpy(np.asarray(s))
               for name, (s, _e, _m, _b) in results.items()}
    sv_path = os.path.join(args.output_dir, "singular_values.pt")
    torch.save(sv_dump, sv_path)
    print(f"[sv] raw singular values -> {sv_path}")

    # ── figures ────────────────────────────────────────────────
    n_blocks = max(n_blocks, 1)
    plot_spectrum(results, n_blocks,
                  os.path.join(args.output_dir, "spectrum_decay.png"))
    plot_energy(results, n_blocks,
                os.path.join(args.output_dir, "energy_retention.png"),
                thresholds=args.energy_thresholds)
    plot_bars(rows, os.path.join(args.output_dir, "per_layer_bars.png"))
    plot_heatmap(rows, "cond", "Condition number (block x layer-type)",
                 os.path.join(args.output_dir, "heatmap_cond.png"))
    plot_heatmap(rows, "eff_rank", "Effective rank (block x layer-type)",
                 os.path.join(args.output_dir, "heatmap_eff_rank.png"))

    print(f"[done] outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
