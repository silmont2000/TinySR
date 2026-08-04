#!/usr/bin/env python3
"""Quantization-sensitivity analysis for TinySR Linear layers.

Implements two calibration-based sensitivity diagnostics on the INPUT
activation of every nn.Linear (the tensor an A4 quantizer would quantize):

  Method 3 - Outlier statistics (activation tail-heaviness):
      * Max / P99.9 ratio of |x|      (tensor-wide)
      * excess kurtosis of x           (tensor-wide)
      * absmax
    Tail-heavy activations => quantization-sensitive.

  Method 2b - Diagonal-Hessian salience (SliM-LLM / GPTQ, diagonal approx):
      * H_jj = E[x_j^2]                (per input channel)
      * salience map  S_ij = w_ij^2 * H_jj
      * layer salience total  = sum_ij S_ij  (= E||Wx||^2 under diagonal approx)
      * salience concentration (top-1% weights / channels share)
    Plus a cheap derived predictor:
      * predicted weight-quant output MSE using actual per-group int4 ranges,
        activation-weighted: sum_ij (quant_var of w_ij) * H_jj.

Outputs a per-layer CSV plus a small JSON summary. Weights are read from the
loaded backbone; only ONE activation forward pass over the calibration images
is required.

The pure-math helpers (StreamingActStats, moment/salience functions) are
importable and covered by test/test_quant_sensitivity.py.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_BACKBONE = (
    "/Users/xieboyang/Documents/Tinysr/可以保留的数据/好/"
    "sweep_tier-2-3/sweep_tier-2-3/merged_backbone"
)


# ---------------------------------------------------------------------------
# pure-math helpers (unit-tested)
# ---------------------------------------------------------------------------

def moment_stats(s1: float, s2: float, s3: float, s4: float, n: int):
    """Return (mean, var, excess_kurtosis) from raw power sums s_k = sum(x^k)."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    m1 = s1 / n
    m2 = s2 / n
    m3 = s3 / n
    m4 = s4 / n
    var = m2 - m1 * m1
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4
    if var <= 0:
        return m1, var, 0.0
    excess_kurt = mu4 / (var * var) - 3.0
    return m1, var, excess_kurt


def outlier_ratio(absmax: float, p999: float, eps: float = 1e-12) -> float:
    """Max / P99.9 ratio; >1, larger => heavier tail."""
    return float(absmax / max(p999, eps))


def layer_salience(W: torch.Tensor, h: torch.Tensor, top_frac: float = 0.01):
    """Diagonal-Hessian salience of a Linear weight.

    W: (out, in) weight.  h: (in,) = E[x_j^2] per input channel.
    Returns dict with total salience, mean-per-weight, and concentration
    (top-`top_frac` share over individual weights and over input channels).
    """
    W = W.detach().float()
    h = h.detach().float().reshape(1, -1)
    S = (W * W) * h                       # (out, in) salience map
    total = float(S.sum())
    numel = S.numel()
    flat = S.reshape(-1)
    k = max(1, int(numel * top_frac))
    top_w_share = float(flat.topk(k).values.sum() / max(total, 1e-30))

    col = S.sum(dim=0)                     # (in,) per-input-channel salience
    kc = max(1, int(col.numel() * top_frac))
    top_ch_share = float(col.topk(kc).values.sum() / max(total, 1e-30))

    return {
        "salience_total": total,
        "salience_mean": total / max(numel, 1),
        "salience_top1pct_w": top_w_share,
        "salience_top1pct_ch": top_ch_share,
    }


def predicted_wquant_out_mse(W: torch.Tensor, h: torch.Tensor,
                             group_size: int = 64, bits: int = 4):
    """Activation-weighted predicted output MSE from per-group int4 weight quant.

    Per group g (along input dim): symmetric int4 step Delta = absmax_g / (2^(b-1)-1),
    uniform quant error variance ~ Delta^2 / 12 for every weight in the group.
    Predicted diagonal output MSE = sum_ij errvar_ij * H_jj.
    Returns (mse, mse_relative) where relative = mse / E||Wx||^2 (diag).
    """
    W = W.detach().float()
    out_f, in_f = W.shape
    qmax = (2 ** (bits - 1)) - 1
    if in_f % group_size != 0:
        # fall back to a single group
        group_size = in_f
    G = in_f // group_size
    Wg = W.reshape(out_f, G, group_size)
    absmax = Wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)  # (out,G,1)
    delta = absmax / qmax
    errvar = (delta * delta) / 12.0                                # (out,G,1)
    errvar_full = errvar.expand(out_f, G, group_size).reshape(out_f, in_f)
    h = h.detach().float().reshape(1, -1)
    mse = float((errvar_full * h).sum())
    signal = float(((W * W) * h).sum())
    return mse, mse / max(signal, 1e-30)


class StreamingActStats:
    """Streaming per-channel and tensor-wide activation statistics.

    Accumulates, over chunks of shape (N, C):
      * per-channel sum of squares  -> E[x_j^2] (Hessian diagonal)
      * tensor-wide power sums s1..s4 (float64) -> mean/var/kurtosis
      * exact abs-max
      * a capped random reservoir of |x| for a P99.9 estimate
    """

    def __init__(self, num_channels: int, reservoir_cap: int = 300_000,
                 seed: int = 0):
        self.C = num_channels
        self.sumsq_ch = torch.zeros(num_channels, dtype=torch.float64)
        self.n_tokens = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.s4 = 0.0
        self.n_elem = 0
        self.absmax = 0.0
        self.reservoir_cap = reservoir_cap
        self._res = torch.empty(0, dtype=torch.float32)
        self._gen = torch.Generator().manual_seed(seed)

    def update(self, x: torch.Tensor):
        xc = x.detach().reshape(-1, self.C).float().cpu().to(torch.float64)
        self.sumsq_ch += (xc * xc).sum(dim=0)
        self.n_tokens += xc.shape[0]
        flat = xc.reshape(-1)
        self.s1 += float(flat.sum())
        self.s2 += float((flat ** 2).sum())
        self.s3 += float((flat ** 3).sum())
        self.s4 += float((flat ** 4).sum())
        self.n_elem += flat.numel()
        a = flat.abs()
        if a.numel():
            self.absmax = max(self.absmax, float(a.max()))
        self._reservoir_add(a.to(torch.float32))

    def _reservoir_add(self, vals: torch.Tensor):
        cap = self.reservoir_cap
        if vals.numel() > cap:
            idx = torch.randperm(vals.numel(), generator=self._gen)[:cap]
            vals = vals[idx]
        merged = torch.cat([self._res, vals])
        if merged.numel() > cap:
            idx = torch.randperm(merged.numel(), generator=self._gen)[:cap]
            merged = merged[idx]
        self._res = merged

    def e_x2(self) -> torch.Tensor:
        """H_jj = E[x_j^2] per channel."""
        return self.sumsq_ch / max(self.n_tokens, 1)

    def finalize(self, p: float = 0.999) -> dict:
        mean, var, kurt = moment_stats(self.s1, self.s2, self.s3, self.s4,
                                       self.n_elem)
        if self._res.numel():
            p999 = float(torch.quantile(self._res, p))
        else:
            p999 = 0.0
        return {
            "act_mean": mean,
            "act_std": var ** 0.5 if var > 0 else 0.0,
            "act_absmax": self.absmax,
            "act_p999": p999,
            "act_max_over_p999": outlier_ratio(self.absmax, p999),
            "act_excess_kurtosis": kurt,
            "n_tokens": self.n_tokens,
        }


# ---------------------------------------------------------------------------
# model-driven analysis (not unit-tested; needs weights + images)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Method 3 (outliers) + 2b (diag-Hessian salience) sensitivity")
    p.add_argument("--pretrained", type=str, default=DEFAULT_BACKBONE)
    p.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    p.add_argument("--input_dir", "-i", type=str, required=True)
    p.add_argument("--output_dir", "-o", type=str, default="outputs/sensitivity")
    p.add_argument("--filter", type=str, default="",
                   help="Only layers whose name contains this substring")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--process_size", type=int, default=512)
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--reservoir_cap", type=int, default=300_000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    return p.parse_args()


def main():
    from utils.device import get_optimal_device_name
    from torchvision import transforms
    from tqdm import tqdm
    from models.tinysr.tinysd3 import TinySD3Transformer2DModel
    from models.vae.autoencoder_tiny import AutoencoderTiny
    from models.pipeline import get_image_names, image_to_latent

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device or get_optimal_device_name())
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[load] transformer <- {args.pretrained}")
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained, torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, cache_dir=args.cache_dir,
    ).to(device, dtype=weight_dtype).eval()
    print(f"[load] vae <- {args.vae_path}")
    vae = AutoencoderTiny.from_pretrained(
        args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir,
    ).to(device, dtype=weight_dtype).eval()

    # collect target Linear layers
    linears = {}
    for name, m in transformer.named_modules():
        if isinstance(m, torch.nn.Linear) and (not args.filter or args.filter in name):
            linears[name] = m
    names = sorted(linears)
    print(f"[layers] {len(names)} Linear"
          + (f" (filter='{args.filter}')" if args.filter else ""))

    stats = {n: StreamingActStats(linears[n].in_features,
                                  reservoir_cap=args.reservoir_cap)
             for n in names}
    handles = []
    for n in names:
        def _mk(nm):
            def _hook(module, inputs):
                stats[nm].update(inputs[0])
            return _hook
        handles.append(linears[n].register_forward_pre_hook(_mk(n)))

    image_names = get_image_names(args.input_dir)
    if args.max_images > 0:
        image_names = image_names[:args.max_images]
    if not image_names:
        raise SystemExit(f"No images in {args.input_dir}")
    print(f"[images] {len(image_names)}")

    tensor_transform = transforms.Compose([transforms.ToTensor()])
    pooled = torch.zeros(1, transformer.config.pooled_projection_dim,
                         device=device, dtype=weight_dtype)
    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)

    with torch.no_grad():
        for img in tqdm(image_names, desc="[calib]"):
            latent, _ = image_to_latent(args.upscale, args.process_size, vae,
                                        img, tensor_transform, device, weight_dtype)
            transformer(hidden_states=latent.to(device, weight_dtype),
                        timestep=timesteps, pooled_projections=pooled,
                        return_dict=False)
    for h in handles:
        h.remove()

    rows = []
    for n in names:
        st = stats[n]
        h = st.e_x2()                              # H_jj
        act = st.finalize()
        W = linears[n].weight.data.cpu()
        sal = layer_salience(W, h)
        pred_mse, pred_rel = predicted_wquant_out_mse(W, h, args.group_size)
        row = {"layer": n, "shape": f"{W.shape[0]}x{W.shape[1]}"}
        row.update(act)
        row.update(sal)
        row["pred_wq_out_mse"] = pred_mse
        row["pred_wq_out_nmse"] = pred_rel
        rows.append(row)

    keys = ["layer", "shape", "act_absmax", "act_p999", "act_max_over_p999",
            "act_excess_kurtosis", "act_std", "n_tokens",
            "salience_total", "salience_mean", "salience_top1pct_w",
            "salience_top1pct_ch", "pred_wq_out_mse", "pred_wq_out_nmse"]
    csv_path = os.path.join(args.output_dir, "quant_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {len(rows)} layers -> {csv_path}")

    # also dump per-channel H (E[x^2]) for downstream group-wise salience
    h_dump = {n: stats[n].e_x2().float() for n in names}
    torch.save(h_dump, os.path.join(args.output_dir, "hessian_diag_ex2.pt"))

    summary = {
        "n_layers": len(rows),
        "images": image_names,
        "top5_outlier_ratio": sorted(
            [(r["layer"], r["act_max_over_p999"]) for r in rows],
            key=lambda x: -x[1])[:5],
        "top5_excess_kurtosis": sorted(
            [(r["layer"], r["act_excess_kurtosis"]) for r in rows],
            key=lambda x: -x[1])[:5],
        "top5_pred_wq_nmse": sorted(
            [(r["layer"], r["pred_wq_out_nmse"]) for r in rows],
            key=lambda x: -x[1])[:5],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
