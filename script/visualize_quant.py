#!/usr/bin/env python3
"""Visualize SVDQuant-style distribution analysis for TinySR linear layers.

For each specified layer, generates a 5-panel figure showing:
  (a) |X|       — original input activation distribution
  (b) |W|       — original weight distribution
  (c) |X_hat|   — after smooth-scaling (activation side)
  (d) |W_hat|   — after smooth-scaling (weight side)

Usage:
    python script/visualize_quant.py \\
        --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr \\
        --vae_path checkpoint/vae/separable \\
        --lora_dir checkpoint/tinysr \\
        --input_image ../RealSR/LR/0001.png \\
        --layers transformer_blocks.0.ff.net.2 transformer_blocks.11.ff.net.2 \\
        --alpha 1.0 --svd_rank 32 --output_dir outputs/vis_quant
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add TinySR to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# per-panel y-axis configuration
# ---------------------------------------------------------------------------
# Set absolute ymax for each panel. None = auto (1.05 × data max).
# Example: {"a": (0, 2.0), "b": (0, 1.5), "c": None, "d": None, "e": (0, 0.3)}
YLIM_CONFIG = {"a": None, "b": (0,0.4), "c": None, "d": (0,1.0), "e": (0,0.4)}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compute_smooth_scale(act_absmax, weight_absmax, alpha, eps=1e-8):
    s = act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
    return s.clamp_min(eps)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _per_channel_percentiles(data: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 50%, 99%, and max across the batch dimension for each channel.

    Parameters
    ----------
    data: (N, C) — activations (N tokens × C channels) or weights (C_out × C_in)

    Returns
    -------
    p50, p99, pmax  each of shape (C,)
    """
    x = _to_np(data.abs())
    p50 = np.percentile(x, 50, axis=0)
    p99 = np.percentile(x, 99, axis=0)
    pmax = x.max(axis=0)
    return p50, p99, pmax


def _plot_fill(ax, p50, p99, pmax, ylim=None):
    """Fill between percentile curves."""
    n = len(p50)
    x = np.arange(n)
    ax.fill_between(x, 0, p50, alpha=1, color="#4dabf7", label="50% pct", linewidth=0)
    ax.fill_between(x, p50, p99, alpha=1, color="#ff922b", label="99% pct", linewidth=0)
    ax.fill_between(x, p99, pmax, alpha=1, color="#e03131", label="Max", linewidth=0)
    ax.set_xlim(0, n - 1)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=6, loc="upper right")


# ---------------------------------------------------------------------------
# layer input capture
# ---------------------------------------------------------------------------

class InputCaptureHook:
    """Capture raw inputs to a module's forward() call."""
    def __init__(self):
        self.inputs = []

    def __call__(self, module, args, kwargs=None):
        self.inputs.append(args[0].detach().cpu())


# ---------------------------------------------------------------------------
# plotter
# ---------------------------------------------------------------------------

def plot_layer(layer_name: str, x_in: torch.Tensor, weight: torch.Tensor,
               svd_rank: int, alpha: float, output_dir: str, no_plot: bool = False) -> dict:
    """Create the 5-panel SVDQuant figure for one layer.  All compute on CPU.
    
    Returns dict of per-layer metrics for CSV dump.
    """
    # Downsample activations if too many tokens (cap at 4096, sample uniformly)
    N, C = x_in.shape
    max_tokens = 2048
    if N > max_tokens:
        idx = torch.linspace(0, N - 1, max_tokens, dtype=torch.long)
        x_in = x_in[idx]

    weight = weight.float()
    N, C = x_in.shape

    # ── per-channel stats ───────────────────────────────────
    w_abs = weight.abs()
    w_absmax = w_abs.amax(dim=0).clamp_min(1e-8)       # (C,)
    act_absmax = x_in.abs().reshape(-1, C).amax(dim=0).clamp_min(1e-8)  # (C,)
    s = compute_smooth_scale(act_absmax, w_absmax, alpha)  # (C,)

    # ── (a) |X| original ────────────────────────────────────
    p50_x, p99_x, pmax_x = _per_channel_percentiles(x_in)

    # ── (b) |W| original ────────────────────────────────────
    p50_w, p99_w, pmax_w = _per_channel_percentiles(weight)

    # ── (c) |X_hat| after smoothing ─────────────────────────
    x_hat = x_in / s.reshape(1, -1)
    p50_xh, p99_xh, pmax_xh = _per_channel_percentiles(x_hat)

    # ── (d) |W_hat| after smoothing ─────────────────────────
    w_hat = weight * s.reshape(1, -1)
    p50_wh, p99_wh, pmax_wh = _per_channel_percentiles(w_hat)

    # ── (e) |R| after SVD (on CPU) ──────────────────────────
    print(f"    svd({w_hat.shape})...")
    u, sv, vh = torch.linalg.svd(w_hat.float(), full_matrices=False)
    rank = min(svd_rank, sv.numel())
    lr = u[:, :rank] @ torch.diag(sv[:rank]) @ vh[:rank, :]
    residual = w_hat - lr
    p50_r, p99_r, pmax_r = _per_channel_percentiles(residual)

    if no_plot:
        # Still need sv_np/cond/eff_rank for metrics
        sv_np = sv.cpu().numpy()
        cond = float(sv_np[0] / max(sv_np[-1], 1e-8))
        eff_rank = float(sv_np.sum() / max(sv_np[0], 1e-8))
        sv_norm = sv_np / sv_np.sum()
        eff_rank_entropy = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-12))))
    else:
        # ── plotting ─────────────────────────────────────────────
        fig, axes = plt.subplots(1, 5, figsize=(22, 3.2))
        fig.suptitle(f"{layer_name}  (α={alpha}, rank={svd_rank})", fontsize=10, y=1.02)

        _plot_fill(axes[0], p50_x, p99_x, pmax_x, ylim=YLIM_CONFIG["a"] or (0, pmax_x.max() * 1.05))
        axes[0].set_title(r"(a) Original, $|X|$  max={:.3f}".format(pmax_x.max()))
        axes[0].set_xlabel("Channel")

        _plot_fill(axes[1], p50_w, p99_w, pmax_w, ylim=YLIM_CONFIG["b"] or (0, pmax_w.max() * 1.05))
        axes[1].set_title(r"(b) Original, $|W|$  max={:.3f}".format(pmax_w.max()))
        axes[1].set_xlabel("Channel")

        _plot_fill(axes[2], p50_xh, p99_xh, pmax_xh, ylim=YLIM_CONFIG["c"] or (0, pmax_xh.max() * 1.05))
        axes[2].set_title(r"(c) After Smoothing, $|\tilde{{X}}| = |X \cdot \operatorname{{diag}}(\lambda)^{{-1}}|$  max={:.3f}".format(pmax_xh.max()))
        axes[2].set_xlabel("Channel")

        _plot_fill(axes[3], p50_wh, p99_wh, pmax_wh, ylim=YLIM_CONFIG["d"] or None)
        axes[3].set_title(r"(d) After Smoothing, $|\tilde{{W}}| = |W \cdot \operatorname{{diag}}(\lambda)|$  max={:.3f}".format(pmax_wh.max()))
        axes[3].set_xlabel("Channel")

        _plot_fill(axes[4], p50_r, p99_r, pmax_r, ylim=YLIM_CONFIG["e"] or (0, pmax_r.max() * 1.05))
        axes[4].set_title(r"(e) After SVD, $|R| = |\tilde{{W}} - L_1 L_2|$  max={:.3f}".format(pmax_r.max()))
        axes[4].set_xlabel("Channel")

        sv_np = sv.cpu().numpy()
        cond = float(sv_np[0] / max(sv_np[-1], 1e-8))
        eff_rank = float(sv_np.sum() / max(sv_np[0], 1e-8))
        sv_norm = sv_np / sv_np.sum()
        eff_rank_entropy = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-12))))
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        out_path = os.path.join(output_dir, f"svdq_vis_{safe_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")

    # ── numerical metrics (reuse sv_np from above) ──────────
    w_arr = _to_np(weight)
    x_arr = _to_np(x_in)           # raw (N, C), can be ±
    x_abs = np.abs(x_arr)

    # --- SVD metrics ---
    sv_top32_energy = float(sv_np[:svd_rank].sum() / sv_np.sum())
    sv_anomaly = float(
        (sv_np[0] - sv_np[1:].mean()) / max(sv_np[0], 1e-8)
    )
    sv_decay_ratio = float(sv_np[min(svd_rank, len(sv_np)-1)] / max(sv_np[0], 1e-8))

    # --- activation statistics ---
    act_mean_overall = float(np.mean(x_abs))
    act_kurtosis = float((np.mean((x_arr - x_arr.mean()) ** 4) /
                           max((x_arr.var() ** 2), 1e-16)))  # excess kurtosis
    # Channel-wise correlation (sample 512 channels to stay <100ms)
    n_sample = min(512, C)
    idx = np.linspace(0, C-1, n_sample, dtype=int)
    x_sample = x_arr[:, idx].astype(np.float64)
    x_sample -= x_sample.mean(axis=0, keepdims=True)
    x_sample /= x_sample.std(axis=0, keepdims=True) + 1e-8
    corr = np.corrcoef(x_sample.T)
    n_off = n_sample * (n_sample - 1)
    act_ch_corr = float(np.abs(corr).sum() - n_sample) / max(n_off, 1)
    # Activation energy concentration (top-32 channels / total)
    ch_energy = (x_abs ** 2).sum(axis=0)
    ch_energy.sort()
    act_energy_top32 = float(ch_energy[-32:].sum() / max(ch_energy.sum(), 1e-8))

    return {
        "layer": layer_name,
        "shape": f"{weight.shape[0]}x{weight.shape[1]}",
        "tokens": N,
        # SVD
        "sv_cond_num":        cond,
        "sv_eff_rank":        eff_rank,
        "sv_eff_rank_entropy": eff_rank_entropy,
        "sv_top32_energy":    sv_top32_energy,
        "sv_anomaly":         sv_anomaly,
        "sv_decay_ratio":     sv_decay_ratio,
        # Activation
        "act_absmax":   float(np.max(x_abs)),
        "act_mean":     act_mean_overall,
        "act_kurtosis": act_kurtosis,
        "act_ch_corr":  act_ch_corr,
        "act_energy_top32": act_energy_top32,
    }

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SVDQuant-style distribution visualizer")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank for loading")
    parser.add_argument("--input_images", type=str, nargs="*", default=None,
                        help="Paths to input LR images (e.g. ../RealSR/LR/0001.png ../RealSR/LR/0002.png)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of LR images (uses all *.png files; capped by --max_images)")
    parser.add_argument("--max_images", type=int, default=20,
                        help="Max number of images to aggregate (default 20)")
    parser.add_argument("--layers", type=str, nargs="*", default=None,
                        help="Layer names to analyze (default: all ff.net.0.proj + ff.net.2 for blocks 0-11)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Smooth alpha (default 1.0)")
    parser.add_argument("--svd_rank", type=int, default=32,
                        help="SVD low-rank for residual visualization")
    parser.add_argument("--output_dir", type=str, default="outputs/vis_quant")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip PNG output (metrics/dumps only)")
    parser.add_argument("--dump_metrics", type=str, default=None,
                        help="Optional path to dump per-layer metrics CSV")
    parser.add_argument("--dump_gates", type=str, default=None,
                        help="Optional path to dump gate_mlp/scale_mlp/shift_mlp per-block CSV + plot")
    parser.add_argument("--quant_attention", action="store_true",
                        help="Replace and calibrate attention layers (W4A4) before analyzing FFN")
    parser.add_argument("--calib_images_attn", type=int, default=20,
                        help="Number of calibration images for attention quant (default 20)")
    parser.add_argument("--propagate_error", action="store_true",
                        help="Measure per-block error propagation through remaining network")
    parser.add_argument("--propagate_hf", action="store_true",
                        help="Also compute high-frequency image error (Laplacian 3/5/7)")
    parser.add_argument("--trace_gram", action="store_true",
                        help="Also compute V·V^T Gram loss for to_v layers (token-value subspace distortion)")
    parser.add_argument("--propagate_fft", type=float, default=None,
                        help="FFT low/high cutoff ratio (0-1). E.g. 0.5 = split at 50%% Nyquist. "
                             "Outputs err_fft_low and err_fft_high.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/.cache/huggingface/")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    selected_blocks = {1, 6, 9, 10, 11}  # hardcoded from sweep results, for gate coloring

    # ── load models ─────────────────────────────────────────
    from models.tinysr.tinysd3 import TinySD3Transformer2DModel
    from models.vae.autoencoder_tiny import AutoencoderTiny
    from models.quant.inference import load_models

    print(f"[load] pretrained...")
    from models.pipeline import load_models
    transformer, vae = load_models(
        args.pretrained_model_name_or_path, args.vae_path, args.lora_dir, args.rank,
        args.cache_dir, args.device if torch.cuda.is_available() else "cpu", weight_dtype,
        skip_lora=(args.lora_dir is None))
    if args.lora_dir:
        transformer = transformer.merge_and_unload()
        print(f"  LoRA merged")
    device = next(transformer.parameters()).device

    # ── resolve image list ───────────────────────────────────
    from models.pipeline import image_to_latent
    from torchvision import transforms

    image_paths = []
    if args.input_images:
        image_paths = list(args.input_images)
    elif args.input_dir:
        image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))

    if not image_paths:
        raise SystemExit("Specify --input_images or --input_dir.")

    image_paths = image_paths[:args.max_images]
    print(f"[images] {len(image_paths)} image(s)")

    tensor_transform = transforms.Compose([transforms.ToTensor()])

    # ── resolve layers (default: all FFN blocks 0-11) ────────
    layer_names = args.layers
    if not layer_names:
        layer_names = []
        for b in range(12):
            layer_names.append(f"transformer_blocks.{b}.ff.net.0.proj")
            layer_names.append(f"transformer_blocks.{b}.ff.net.2")

    # ── timestep embedding (same for all images) ─────────────
    pooled_proj = torch.zeros(1, transformer.config.pooled_projection_dim,
                               device=device, dtype=weight_dtype)
    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)

    # ── optionally quantize attention layers first ────────────
    if args.quant_attention:
        from models.quant.layers import (
            get_target_suffixes, replace_linear_with_w4a4,
            set_quant_enabled, set_observer_enabled,
        )
        from models.quant.calibrate import calibrate_all_layers
        from models.quant.inference import build_layer_replacement_kwargs

        attn_suffixes = get_target_suffixes("attn_only")
        qt_kwargs = build_layer_replacement_kwargs(
            w_bits=4, a_bits=4, svdq_rank=32, svdq_smooth_alpha=args.alpha)
        replaced = replace_linear_with_w4a4(
            transformer, target_suffixes=attn_suffixes, skip_keywords=("lora_",),
            **qt_kwargs)
        print(f"[quant_attn] replaced {len(replaced)} attention layers -> QuantLinearW4A4")

        # Phase 1: FP16 forward to collect stats
        set_quant_enabled(transformer, False)
        set_observer_enabled(transformer, True)
        calib_imgs = image_paths[:min(args.calib_images_attn, len(image_paths))]
        for img_path in tqdm(calib_imgs, desc="[attn calib]"):
            latent, _ = image_to_latent(
                4, 512, vae, img_path, tensor_transform, device, weight_dtype)
            latent = latent.to(device=device, dtype=weight_dtype)
            with torch.no_grad():
                transformer(
                    hidden_states=latent, timestep=timesteps,
                    pooled_projections=pooled_proj, return_dict=False)
        set_observer_enabled(transformer, False)

        # Phase 2: freeze attention layers
        calibrate_all_layers(transformer, do_search=False, alpha_grid_size=7)
        set_quant_enabled(transformer, True)
        print(f"[quant_attn] attention layers frozen, FFN layers remain FP16")

    # ── capture activations across multiple images ───────────
    print(f"[capture] layers: {layer_names}")

    hooks = {}
    handles = {}
    for name in layer_names:
        try:
            m = transformer.get_submodule(name)
        except AttributeError:
            print(f"  [WARN] layer not found: {name}")
            continue
        h = InputCaptureHook()
        handles[name] = m.register_forward_pre_hook(h)
        hooks[name] = h

    for img_path in tqdm(image_paths, desc="[capture]"):
        model_input, _ = image_to_latent(
            4, 512, vae, img_path, tensor_transform, device, weight_dtype)
        latent = model_input.to(device=device, dtype=weight_dtype)
        with torch.no_grad():
            transformer(
                hidden_states=latent,
                timestep=timesteps,
                pooled_projections=pooled_proj,
                return_dict=False,
            )

    for name, handle in handles.items():
        handle.remove()

    captured = {}
    for name, h in hooks.items():
        if h.inputs:
            captured[name] = torch.cat(h.inputs, dim=0)
        else:
            print(f"  [WARN] no activations captured for {name}")
    print(f"  captured {len(captured)} layers")
    # ── gate_mlp analysis (per block) ────────────────────────
    if args.dump_gates:
        gate_rows = []
        for b in range(12):
            try:
                block = transformer.transformer_blocks[b]
            except (IndexError, AttributeError):
                continue
            gate  = block.gate_mlp.detach().cpu().float().numpy().ravel()
            scale = block.scale_mlp.detach().cpu().float().numpy().ravel()
            shift = block.shift_mlp.detach().cpu().float().numpy().ravel()
            gate_rows.append({
                "block": b,
                "gate_mean": float(np.mean(gate)),
                "gate_std":  float(np.std(gate)),
                "gate_min":  float(np.min(gate)),
                "gate_max":  float(np.max(gate)),
                "gate_p90":  float(np.percentile(gate, 90)),
                "gate_p10":  float(np.percentile(gate, 10)),
                "gate_spread": float(np.percentile(gate, 90) - np.percentile(gate, 10)),
                "scale_mean": float(np.mean(scale)),
                "scale_std":  float(np.std(scale)),
                "scale_max":  float(np.max(scale)),
                "shift_mean": float(np.mean(shift)),
                "shift_std":  float(np.std(shift)),
            })

        import csv as _csv
        os.makedirs(os.path.dirname(args.dump_gates) or ".", exist_ok=True)
        with open(args.dump_gates, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=gate_rows[0].keys())
            w.writeheader(); w.writerows(gate_rows)
        print(f"[gates] saved {len(gate_rows)} blocks -> {args.dump_gates}")

        # Also save raw per-channel gate_mlp (1536 x 12) for per-channel analysis
        raw_gates = torch.stack([transformer.transformer_blocks[b].gate_mlp.detach().cpu()
                                  for b in range(12)], dim=0)  # (12, 1536)
        raw_path = args.dump_gates.replace(".csv", "_raw.pt")
        torch.save(raw_gates, raw_path)
        print(f"[gates] raw gate_mlp (12x1536) -> {raw_path}")

        # Gate distribution bar plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
        sel_color = {b: ("#e03131" if b in selected_blocks else "#4dabf7") for b in range(12)}
        for b, r in enumerate(gate_rows):
            color = "#e03131" if b in {1,6,9,10,11} else "#4dabf7"
            axes[0].bar(b, r["gate_spread"], color=color, alpha=0.85)
            axes[1].bar(b, r["gate_max"], color=color, alpha=0.85)
            axes[2].bar(b, r["gate_mean"], color=color, alpha=0.85)
        axes[0].set_ylabel("gate spread (p90-p10)"); axes[0].set_title("AdaLayerNormZero gate_mlp per block  (red=selected, blue=unselected)")
        axes[1].set_ylabel("gate_max")
        axes[2].set_ylabel("gate_mean"); axes[2].set_xlabel("Block index")
        axes[2].set_xticks(range(12))
        gate_plot = args.dump_gates.replace(".csv", ".png")
        fig.savefig(gate_plot, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[gates] plot saved -> {gate_plot}")

    # ── error propagation analysis (quantize one layer at a time) ─────
    if args.propagate_error:
        from models.quant.layers import replace_linear_with_w4a4, set_quant_enabled, set_observer_enabled
        from models.quant.calibrate import calibrate_one_layer, compute_smooth_scale
        from models.quant.inference import build_layer_replacement_kwargs

        qt_kw = build_layer_replacement_kwargs(w_bits=4, a_bits=4, svdq_rank=args.svd_rank,
                                                svdq_smooth_alpha=args.alpha)

        # FP16 reference + capture all layer outputs (forward hooks)
        ref_outputs = {}
        def _make_fwd_hook(store):
            def _hook(module, inputs, output):
                store["out"] = output.detach().clone()
            return _hook
        fwd_handles = {}
        for name in layer_names:
            m = transformer.get_submodule(name)
            store = {}
            fwd_handles[name] = (m.register_forward_hook(_make_fwd_hook(store)), store)

        # Hook last block to_v for propagate Gram
        gram_ref_store = {}
        if args.trace_gram:
            gram_ref_h = transformer.transformer_blocks[11].attn.to_v.register_forward_hook(
                _make_fwd_hook(gram_ref_store))

        img0 = image_paths[0]
        latent_ref, _ = image_to_latent(4, 512, vae, img0, tensor_transform, device, weight_dtype)
        latent_ref = latent_ref.to(device, dtype=weight_dtype)
        set_quant_enabled(transformer, False)
        with torch.no_grad():
            y_ref = transformer(hidden_states=latent_ref, timestep=timesteps,
                                pooled_projections=pooled_proj, return_dict=False)[0].clone()
        for h, _ in fwd_handles.values(): h.remove()
        ref_outputs = {name: s["out"] for name, (_, s) in fwd_handles.items()}
        V_ref_last = gram_ref_store.get("out")
        if args.trace_gram:
            gram_ref_h.remove()
        print(f"\n[propagate] FP16 reference (shape={tuple(y_ref.shape)}), "
              f"{len(ref_outputs)} layer outputs captured")

        errors = []
        print(f"{'Layer':<55s} {'err_propagated':>14s} {'err_up':>14s} {'err_down':>14s}")
        print("-" * 100)

        def _save_one(name):
            pn, ca = name.rsplit(".", 1); pd = transformer.get_submodule(pn); om = getattr(pd, ca)
            return (pn, ca, om)

        def _restore_one(pn, ca, om):
            setattr(transformer.get_submodule(pn), ca, om)

        def _calibrate_one(new_m, name):
            set_observer_enabled(transformer, False)
            new_m.weight_quantizer.observer_enabled = True
            new_m.act_quantizer.observer_enabled = True
            with torch.no_grad():
                transformer(hidden_states=latent_ref, timestep=timesteps,
                             pooled_projections=pooled_proj, return_dict=False)
            new_m.weight_quantizer.observer_enabled = False
            new_m.act_quantizer.observer_enabled = False
            calibrate_one_layer(new_m, name, do_search=False, alpha_grid_size=7)
            new_m.weight_quantizer.enabled = True
            new_m.act_quantizer.enabled = True

        for name in tqdm(layer_names, desc="[propagate]"):
            if not isinstance(transformer.get_submodule(name.rsplit(".",1)[0] if "." in name else ""),
                              torch.nn.Module) or name not in ref_outputs:
                continue

            pn, ca, om = _save_one(name)

            # Replace just this ONE layer
            replace_linear_with_w4a4(transformer, target_suffixes=[name],
                                      skip_keywords=("lora_",), **qt_kw)
            new_m = getattr(transformer.get_submodule(pn), ca)
            _calibrate_one(new_m, name)

            # Forward
            q_store = {}
            q_handle = new_m.register_forward_hook(_make_fwd_hook(q_store))
            gram_quant_store = {}
            if args.trace_gram and V_ref_last is not None:
                gram_quant_h = transformer.transformer_blocks[11].attn.to_v.register_forward_hook(
                    _make_fwd_hook(gram_quant_store))
            with torch.no_grad():
                y_quant = transformer(hidden_states=latent_ref, timestep=timesteps,
                                       pooled_projections=pooled_proj, return_dict=False)[0]
            q_handle.remove()
            if args.trace_gram and V_ref_last is not None:
                gram_quant_h.remove()
            err_prop = (y_ref - y_quant).pow(2).mean().item()

            # Gram loss at last block: how much quantizing THIS layer distorts block 11's V·V^T
            err_gram = None
            if args.trace_gram and V_ref_last is not None:
                V_quant_last = gram_quant_store.get("out")
                if V_quant_last is not None and V_ref_last.shape == V_quant_last.shape:
                    Vr = V_ref_last.float().reshape(-1, V_ref_last.shape[-1])
                    Vq = V_quant_last.float().reshape(-1, V_quant_last.shape[-1])
                    Gr = Vr @ Vr.T
                    Gq = Vq @ Vq.T
                    err_gram = (Gr - Gq).pow(2).mean().item()

            # HF + FFT (on y_quant)
            err_hf = {}
            if args.propagate_hf:
                def _lap_kernel(k):
                    w = torch.ones(k, k, dtype=torch.float32, device=device)
                    w[k//2, k//2] = -(k*k - 1)
                    return w.view(1,1,k,k)
                laps = {k: _lap_kernel(k) for k in (3,5,7)}
                def _hf_err_k(latent, lap):
                    img = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
                    return torch.nn.functional.conv2d(img.float(), lap.repeat(3,1,1,1), padding=lap.shape[-1]//2, groups=3)
                hf_ref = {k: _hf_err_k(y_ref, lap) for k, lap in laps.items()}
                hf_quant = {k: _hf_err_k(y_quant, lap) for k, lap in laps.items()}
                for k in (3,5,7):
                    err_hf[k] = (hf_ref[k] - hf_quant[k]).pow(2).mean().item()

            err_fft = {}
            if args.propagate_fft is not None:
                cutoff = args.propagate_fft
                def _fft_bands(img):
                    mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(img.float())))
                    H, W = mag.shape[-2:]; cy, cx = H//2, W//2
                    y, x = torch.meshgrid(torch.arange(H, device=img.device),
                                           torch.arange(W, device=img.device), indexing='ij')
                    r = torch.sqrt((y-cy).float()**2 + (x-cx).float()**2)
                    thresh = r.max() * cutoff
                    mask_low = (r < thresh).float().unsqueeze(0).unsqueeze(0)
                    mask_high = (r >= thresh).float().unsqueeze(0).unsqueeze(0)
                    el = (mag * mask_low).sum() / mask_low.sum().clamp_min(1)
                    eh = (mag * mask_high).sum() / mask_high.sum().clamp_min(1)
                    return el.item(), eh.item()
                im_ref = vae.decode(y_ref / vae.config.scaling_factor, return_dict=False)[0]
                im_quant = vae.decode(y_quant / vae.config.scaling_factor, return_dict=False)[0]
                low_r, high_r = _fft_bands(im_ref)
                low_q, high_q = _fft_bands(im_quant)
                err_fft["low"] = (low_r - low_q)**2
                err_fft["high"] = (high_r - high_q)**2

            _restore_one(pn, ca, om)

            ep_str = f"{err_prop:.4e}" if not math.isnan(err_prop) else "NaN"
            tqdm.write(f"  {name:<53s} {ep_str:>14s}")

            err_entry = {"layer": name, "err_propagated": err_prop}
            if args.trace_gram and err_gram is not None:
                err_entry["err_gram"] = err_gram
            if args.propagate_hf:
                for k in (3,5,7): err_entry[f"err_hf_{k}"] = err_hf.get(k)
            if args.propagate_fft is not None:
                err_entry["err_fft_low"] = err_fft.get("low")
                err_entry["err_fft_high"] = err_fft.get("high")
            errors.append(err_entry)

        # Save CSV
        if args.dump_metrics:
            epath = args.dump_metrics.replace(".csv", "_propagate.csv")
            os.makedirs(os.path.dirname(epath) or ".", exist_ok=True)
            import csv as _csv
            with open(epath, "w", newline="") as f:
                fnames = errors[0].keys()
                w = _csv.DictWriter(f, fieldnames=fnames)
                w.writeheader(); w.writerows(errors)
            print(f"[propagate] saved {len(errors)} layers -> {epath}")
        print()

    # ── free GPU memory before plotting ─────────────────────
    del model_input, pooled_proj, timesteps, image_paths
    transformer = transformer.cpu()
    vae = vae.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  GPU memory freed, plotting on CPU")

    # ── plot each layer (on CPU to avoid OOM) ───────────────
    all_metrics = []
    for name in layer_names:
        if name not in captured:
            print(f"  [SKIP] no activation data for {name}")
            continue
        try:
            module = transformer.get_submodule(name)
        except AttributeError:
            print(f"  [SKIP] cannot find module: {name}")
            continue

        if not hasattr(module, "weight"):
            print(f"  [SKIP] {name} has no weight attribute")
            continue

        x_in = captured[name]
        weight = module.weight.data.cpu()

        # Flatten activation to (N, in_features)
        x_in = torch.as_tensor(x_in).reshape(-1, x_in.shape[-1])
        N, C = x_in.shape

        print(f"  [plot] {name}  x=({N},{C})  w=({weight.shape[0]},{weight.shape[1]})")

        if C != weight.shape[1]:
            print(f"  [SKIP] dimension mismatch: x_in[-1]={C} vs w.in_features={weight.shape[1]}")
            continue

        metrics = plot_layer(name, x_in, weight, args.svd_rank, args.alpha,
                              args.output_dir, no_plot=args.no_plot)
        all_metrics.append(metrics)

    # ── dump CSV ─────────────────────────────────────────────
    if args.dump_metrics and all_metrics:
        import csv
        os.makedirs(os.path.dirname(args.dump_metrics) or ".", exist_ok=True)
        keys = all_metrics[0].keys()
        with open(args.dump_metrics, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_metrics)
        print(f"[metrics] {len(all_metrics)} layers -> {args.dump_metrics}")

    print(f"[done] outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
