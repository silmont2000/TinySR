#!/usr/bin/env python3
"""SVDQuant smoothed-weight spectrum: cond / eff_rank of W_hat = W * diag(s).

Reproduces the metric used by visualize_quant.py: the SVD is taken on the
activation-SMOOTHED weight, not the raw weight, where

    s_j   = act_absmax_j^alpha / w_absmax_j^(1-alpha)      (per input channel)
    W_hat = W * s          (scales input columns)

cond    = sigma_max / sigma_min
eff_rk  = sum(sigma) / sigma_max          (nuclear / spectral, as in visualize_quant)
stable  = sum(sigma^2) / sigma_max^2
top32   = sum(sigma[:32]^2) / sum(sigma^2)   (rank-32 energy)

Per-channel activation absmax is captured in ONE calibration forward pass.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_BASE = "checkpoint/tinybackbone/prune-12-merge-tinysr/transformer"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=DEFAULT_BASE)
    p.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    p.add_argument("--input_dir", "-i", type=str, default="dataset/test_image")
    p.add_argument("--output_dir", "-o", type=str, default="outputs/ffnet_tables")
    p.add_argument("--filter", type=str, default="ff.net.2")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--svd_rank", type=int, default=32)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--process_size", type=int, default=512)
    p.add_argument("--tag", type=str, default="base")
    return p.parse_args()


class PerChannelAbsmax:
    def __init__(self, C):
        self.absmax = torch.zeros(C, dtype=torch.float32)

    def update(self, x):
        xc = x.detach().reshape(-1, self.absmax.shape[0]).abs().float().cpu()
        self.absmax = torch.maximum(self.absmax, xc.amax(dim=0))


def smoothed_metrics(W, act_absmax, alpha, svd_rank):
    W = W.float()
    w_absmax = W.abs().amax(dim=0).clamp_min(1e-8)          # (in,)
    s = (act_absmax.clamp_min(1e-8).pow(alpha) /
         w_absmax.pow(1.0 - alpha)).clamp_min(1e-8)         # (in,)
    W_hat = W * s.reshape(1, -1)
    sv = torch.linalg.svdvals(W_hat.float())
    s0 = sv[0].item()
    cond = s0 / max(sv[-1].item(), 1e-8)
    eff_rk = sv.sum().item() / max(s0, 1e-8)
    stable = (sv ** 2).sum().item() / max(s0 ** 2, 1e-8)
    r = min(svd_rank, sv.numel())
    top_energy = (sv[:r] ** 2).sum().item() / (sv ** 2).sum().item()
    return {
        "cond": cond, "eff_rk": eff_rk, "stable_rank": stable,
        "top32_energy": top_energy,
        "w_hat_max": float(W_hat.abs().max()),
        "act_max": float(act_absmax.max()),
    }


def main():
    from utils.device import get_optimal_device_name
    from torchvision import transforms
    from tqdm import tqdm
    from models.tinysr.tinysd3 import TinySD3Transformer2DModel
    from models.vae.autoencoder_tiny import AutoencoderTiny
    from models.pipeline import get_image_names, image_to_latent

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(get_optimal_device_name())
    wd = torch.float16 if device.type == "cuda" else torch.float32

    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained, torch_dtype=wd, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True).to(device, wd).eval()
    vae = AutoencoderTiny.from_pretrained(
        args.vae_path, torch_dtype=wd).to(device, wd).eval()

    linears = {n: m for n, m in transformer.named_modules()
               if isinstance(m, torch.nn.Linear)
               and (not args.filter or args.filter in n)}
    names = sorted(linears)
    caps = {n: PerChannelAbsmax(linears[n].in_features) for n in names}
    handles = [linears[n].register_forward_pre_hook(
        (lambda nm: (lambda mod, inp: caps[nm].update(inp[0])))(n)) for n in names]

    imgs = get_image_names(args.input_dir)
    if args.max_images > 0:
        imgs = imgs[:args.max_images]
    tf = transforms.Compose([transforms.ToTensor()])
    pooled = torch.zeros(1, transformer.config.pooled_projection_dim, device=device, dtype=wd)
    ts = torch.tensor([1000.0], device=device, dtype=wd)
    print(f"[calib] {len(imgs)} images, {len(names)} layers (filter='{args.filter}')")
    with torch.no_grad():
        for im in tqdm(imgs, desc="[calib]"):
            lat, _ = image_to_latent(args.upscale, args.process_size, vae, im, tf, device, wd)
            transformer(hidden_states=lat.to(device, wd), timestep=ts,
                        pooled_projections=pooled, return_dict=False)
    for h in handles:
        h.remove()

    rows = []
    for n in names:
        W = linears[n].weight.data.cpu()
        m = smoothed_metrics(W, caps[n].absmax, args.alpha, args.svd_rank)
        bn = int(n.split("blocks.")[1].split(".")[0])
        m = {"layer": n.replace("transformer_blocks.", "b"), "block": bn, **m}
        rows.append(m)
    rows.sort(key=lambda r: r["block"])

    out = os.path.join(args.output_dir, f"smoothed_cond_{args.tag}.csv")
    keys = ["layer", "block", "cond", "eff_rk", "stable_rank", "top32_energy",
            "w_hat_max", "act_max"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'block':>5} {'cond':>8} {'eff_rk':>8} {'top32E':>8} {'w_hat_mx':>9} {'act_max':>8}")
    for r in rows:
        print(f"b{r['block']:<4} {r['cond']:8.1f} {r['eff_rk']:8.1f} "
              f"{r['top32_energy']:8.3f} {r['w_hat_max']:9.2f} {r['act_max']:8.2f}")
    print(f"[csv] -> {out}")


if __name__ == "__main__":
    main()
