#!/usr/bin/env python3
"""
Per-layer quantization sensitivity analysis.

For each quantized linear layer, measures the reconstruction error when
quantized to W4A8, sweeping smooth_alpha to find the best-case error.
Layers are ranked by output MSE to identify those most sensitive to quantization.

Usage:
    python script/sensitivity_analysis.py \
        --quant_scope ffn_only \
        --calib_images 8 \
        --output_dir outputs/sensitivity_ffn

Outputs:
    - sensitivity_report.json  : full per-layer metrics (weight/activation stats, per-alpha errors)
    - sensitivity_ranking.txt  : human-readable ranked list
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.quant.layers import (
    get_target_suffixes,
    iter_quant_layers,
    replace_linear_with_w4a4,
    set_observer_enabled,
    set_quant_enabled,
)
from models.quant.ops import affine_fake_quant_weight, fake_quant_activation
from models.quant.inference import (
    get_image_names,
    get_weight_dtype,
    image_to_latent,
    load_models,
)
from models.quant.tiler import tile_sample


# ---------------------------------------------------------------------------
# Weight / activation statistics
# ---------------------------------------------------------------------------

def compute_weight_stats(weight: torch.Tensor) -> dict:
    w = weight.detach().float()
    w_mean = w.mean(dim=1, keepdim=True)
    w_std = w.std(dim=1)
    w_centered = w - w_mean
    w_kurt = (w_centered ** 4).mean(dim=1) / (w_std ** 4 + 1e-12) - 3.0
    outlier_ratio = (w_centered.abs() > 5 * w_std.unsqueeze(1)).float().mean(dim=1)
    return {
        "std_mean": float(w_std.mean().item()),
        "std_max": float(w_std.max().item()),
        "kurtosis_mean": float(w_kurt.mean().item()),
        "kurtosis_max": float(w_kurt.max().item()),
        "outlier_ratio_mean": float(outlier_ratio.mean().item()),
        "outlier_ratio_max": float(outlier_ratio.max().item()),
    }


def compute_activation_stats(inputs: torch.Tensor) -> dict:
    if inputs is None or inputs.numel() == 0:
        return {"abs_max": 0.0, "std": 0.0, "abs_mean": 0.0}
    x = inputs.float()
    return {
        "abs_max": float(x.abs().max().item()),
        "std": float(x.std().item()),
        "abs_mean": float(x.abs().mean().item()),
    }


# ---------------------------------------------------------------------------
# Per-layer sensitivity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_layer_sensitivity(
    layer_name: str,
    weight: torch.Tensor,
    act_absmax: torch.Tensor,
    input_cache: list,
    weight_bits: int,
    act_bits: int,
    alphas: list,
    device: torch.device,
) -> dict:
    """Evaluate output reconstruction error for a single layer across smooth_alphas."""

    out_features, in_features = weight.shape
    weight = weight.to(device=device, dtype=torch.float32)
    weight_absmax = weight.abs().amax(dim=0).clamp_min(1e-8)

    if input_cache and len(input_cache) > 0:
        inputs_cat = torch.cat(input_cache, dim=0).to(device=device, dtype=torch.float32)
        act_stats = compute_activation_stats(inputs_cat)
    else:
        inputs_cat = None
        act_stats = compute_activation_stats(None)

    w_stats = compute_weight_stats(weight)

    if act_absmax is not None:
        act_absmax = act_absmax.to(device=device, dtype=torch.float32).clamp_min(1e-8)

    alpha_results = []
    best_output_mse = float("inf")
    best_alpha = 0.5
    best_snr = None

    for alpha in alphas:
        if act_absmax is not None:
            smooth_scale = act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
            smooth_scale = smooth_scale.clamp_min(1e-8)
        else:
            smooth_scale = weight_absmax.pow(alpha - 1.0).clamp_min(1e-8)

        smoothed_weight = weight * smooth_scale.reshape(1, -1)

        # Weight-only MSE
        w_q = affine_fake_quant_weight(smoothed_weight, bits=weight_bits, symmetric=True)
        w_mse = (smoothed_weight - w_q).pow(2).mean().item()
        w_mse_rel = w_mse / (smoothed_weight.pow(2).mean().item() + 1e-12)

        # Output MSE (requires calibration inputs)
        output_mse = None
        output_mse_rel = None
        snr_db = None

        if inputs_cat is not None:
            inputs_smoothed = inputs_cat / smooth_scale.reshape(1, -1)
            orig_out = inputs_cat @ weight.T

            q_inputs = fake_quant_activation(inputs_smoothed, bits=act_bits, symmetric=True)
            q_out = q_inputs @ w_q.T

            output_mse = (orig_out - q_out).pow(2).mean().item()
            signal_power = orig_out.pow(2).mean().item()
            output_mse_rel = output_mse / (signal_power + 1e-12)
            snr_db = float(10.0 * math.log10(signal_power / output_mse)) if output_mse > 1e-12 else float("inf")

            if output_mse < best_output_mse:
                best_output_mse = output_mse
                best_alpha = alpha
                best_snr = snr_db

        alpha_results.append({
            "alpha": alpha,
            "weight_mse": w_mse,
            "weight_mse_rel": w_mse_rel,
            "output_mse": output_mse,
            "output_mse_rel": output_mse_rel,
            "snr_db": snr_db,
        })

    return {
        "layer_name": layer_name,
        "in_features": in_features,
        "out_features": out_features,
        "best_alpha": best_alpha,
        "best_output_mse": best_output_mse,
        "best_snr_db": best_snr,
        "weight_stats": w_stats,
        "activation_stats": act_stats,
        "alpha_results": alpha_results,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_calibration(transformer, calib_data_list, forward_fn):
    set_quant_enabled(transformer, False)
    set_observer_enabled(transformer, True)

    for model_input, timesteps, pooled_prompt_embeds, weight_dtype in tqdm(
        calib_data_list, desc="Calibration"):
        forward_fn(model_input, timesteps, pooled_prompt_embeds, weight_dtype)
        if model_input.device.type == "cuda":
            torch.cuda.empty_cache()

    set_observer_enabled(transformer, False)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _make_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-layer W4A8 quantization sensitivity analysis")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank for loading")

    # Data
    parser.add_argument("--calib_input_dir", type=str,
                        default="dataset/StableSR_testsets/DrealSRVal_crop128/test_LR")
    parser.add_argument("--calib_images", type=int, default=8)

    # Quantization
    parser.add_argument("--quant_scope", type=str, default="ffn_only",
                        choices=["ffn_only", "attn_only", "dit_full"])
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--svdq_rank", type=int, default=0,
                        help="SVDQ branch rank (0 = affine-only, no branch). "
                             "Set >0 to test with low-rank compensation.")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--skip_lora", action="store_true",
                        help="Skip LoRA loading for faster iteration")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/sensitivity")
    parser.add_argument("--alpha_grid_size", type=int, default=11,
                        help="Number of grid points for smooth_alpha sweep (2-21)")

    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.mixed_precision)

    print(f"[SENS] device={device}, dtype={weight_dtype}")
    print(f"[SENS] scope={args.quant_scope}, w{args.w_bits}a{args.a_bits}, svdq_rank={args.svdq_rank}")
    print(f"[SENS] calib_images={args.calib_images}")

    # ---- Load models ----
    print("[SENS] Loading models ...")
    transformer, vae = load_models(
        args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
        args.rank, args.cache_dir, device, weight_dtype,
        skip_lora=args.skip_lora)

    # ---- Replace linear layers with QuantLinearW4A4 ----
    # Use LowRankAffineQuantComponent (svdq kind) even when rank=0, because
    # AffineQuantComponent does NOT collect input_cache / act_absmax needed
    # for sensitivity analysis. With rank=0, the SVDQ component behaves like
    # affine (no branch built) but retains input collection capability.
    target_suffixes = get_target_suffixes(args.quant_scope)

    w_kwargs = {
        "bits": args.w_bits, "symmetric": True, "per_channel": True, "ch_axis": 0,
        "rank": args.svdq_rank, "smooth_alpha": 0.5, "max_gptq_samples": 2048,
    }
    a_kwargs = {
        "bits": args.a_bits, "symmetric": True, "per_channel": False,
    }

    replaced = replace_linear_with_w4a4(
        transformer,
        target_suffixes=target_suffixes,
        skip_keywords=("lora_",),
        weight_quant_kind="svdq",
        weight_quant_kwargs=w_kwargs,
        act_quant_kwargs=a_kwargs,
    )
    print(f"[SENS] Replaced {len(replaced)} layers with QuantLinearW4A4")

    # ---- Prepare calibration data ----
    calib_image_names = get_image_names(args.calib_input_dir)
    calib_count = min(args.calib_images, len(calib_image_names))
    if calib_count == 0:
        raise RuntimeError(f"No calibration images in {args.calib_input_dir}")

    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location="cpu",
    ).to(device=device, dtype=weight_dtype)
    timesteps = torch.tensor([args.timestep], device=device, dtype=weight_dtype)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    calib_data_list = []
    for image_path in tqdm(calib_image_names[:calib_count], desc="Preparing calib"):
        model_input, _ = image_to_latent(
            args.upscale, args.process_size, vae, image_path,
            tensor_transform, device, weight_dtype)
        calib_data_list.append(
            (model_input, timesteps, pooled_prompt_embeds, weight_dtype))

    def _forward_fn(model_input, ts, ppe, wd):
        tile_sample(model_input, transformer, ts, ppe, wd,
                    latent_tiled_size=args.latent_tiled_size,
                    latent_tiled_overlap=args.latent_tiled_overlap)

    # ---- Calibration (collects input_cache + act_absmax) ----
    print(f"[SENS] Running calibration ({calib_count} images) ...")
    run_calibration(transformer, calib_data_list, _forward_fn)

    # ---- Per-layer sensitivity evaluation ----
    num_grids = max(2, min(args.alpha_grid_size, 21))
    alphas = [i / (num_grids - 1) for i in range(num_grids)]
    print(f"[SENS] Evaluating sensitivity with alpha grid: {[f'{a:.2f}' for a in alphas]}")

    results = []
    quant_layers = list(iter_quant_layers(transformer))
    for name, m in tqdm(quant_layers, desc="Evaluating layers"):
        weight = m.weight.detach()
        act_absmax = getattr(m.weight_quantizer, "act_absmax", None)
        input_cache = getattr(m.weight_quantizer, "input_cache", None) or []

        result = evaluate_layer_sensitivity(
            layer_name=name,
            weight=weight,
            act_absmax=act_absmax,
            input_cache=input_cache,
            weight_bits=args.w_bits,
            act_bits=args.a_bits,
            alphas=alphas,
            device=device,
        )
        results.append(result)

    # ---- Rank and report ----
    results_with_mse = [r for r in results if r["best_output_mse"] is not None]
    results_without_mse = [r for r in results if r["best_output_mse"] is None]
    results_with_mse.sort(key=lambda r: r["best_output_mse"], reverse=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Print top sensitive layers
    print("\n" + "=" * 95)
    print("TOP 15 MOST SENSITIVE LAYERS (by best output MSE across alpha sweep)")
    print("=" * 95)
    header = f"{'Rank':<5} {'Layer':<55} {'OutMSE':>10} {'SNR(dB)':>9} {'Alpha':>6}"
    print(header)
    print("-" * 95)
    top_n = min(15, len(results_with_mse))
    for i, r in enumerate(results_with_mse[:top_n]):
        print(f"{i+1:<5} {r['layer_name']:<55} {r['best_output_mse']:>10.6e} "
              f"{r['best_snr_db']:>9.2f} {r['best_alpha']:>6.2f}")

    if len(results_with_mse) > top_n:
        print(f"\n... ({len(results_with_mse) - top_n} more layers) ...\n")
        print("=" * 95)
        print("BOTTOM 5 LEAST SENSITIVE LAYERS")
        print("=" * 95)
        print(header)
        print("-" * 95)
        for i, r in enumerate(results_with_mse[-5:]):
            rank = len(results_with_mse) - 5 + i + 1
            print(f"{rank:<5} {r['layer_name']:<55} {r['best_output_mse']:>10.6e} "
                  f"{r['best_snr_db']:>9.2f} {r['best_alpha']:>6.2f}")

    if results_without_mse:
        print(f"\n[WARN] {len(results_without_mse)} layers had no calibration inputs "
              f"(no output MSE computed)")

    # ---- Save JSON report ----
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "num_layers_total": len(results),
        "num_layers_with_output": len(results_with_mse),
        "layers": results_with_mse,
        "layers_no_input": results_without_mse,
    }
    report = _make_json_safe(report)

    report_path = os.path.join(args.output_dir, "sensitivity_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[SENS] Full report -> {report_path}")

    # ---- Save human-readable ranking ----
    ranking_path = os.path.join(args.output_dir, "sensitivity_ranking.txt")
    with open(ranking_path, "w", encoding="utf-8") as f:
        f.write(f"Layer Sensitivity Ranking (W{args.w_bits}A{args.a_bits}, "
                f"scope={args.quant_scope}, svdq_rank={args.svdq_rank})\n")
        f.write("Ranked by best output MSE (lower SNR = more sensitive)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Rank':<5} {'OutputMSE':>12} {'SNR(dB)':>9} {'Alpha':>6}  "
                f"{'W_Kurt_Mean':>12} {'W_Outlier':>11} {'A_Max':>9}  Layer\n")
        f.write("-" * 100 + "\n")
        for i, r in enumerate(results_with_mse):
            ws = r["weight_stats"]
            a_s = r["activation_stats"]
            f.write(f"{i+1:<5} {r['best_output_mse']:>12.6e} {r['best_snr_db']:>9.2f} "
                    f"{r['best_alpha']:>6.2f}  "
                    f"{ws['kurtosis_mean']:>12.4f} {ws['outlier_ratio_mean']:>11.6f} "
                    f"{a_s['abs_max']:>9.4f}  {r['layer_name']}\n")

        if results_without_mse:
            f.write(f"\n--- Layers without calibration inputs ({len(results_without_mse)}) ---\n")
            for r in results_without_mse:
                f.write(f"  {r['layer_name']}\n")

    print(f"[SENS] Ranking -> {ranking_path}")
    print("[SENS] Done.")


if __name__ == "__main__":
    main()
