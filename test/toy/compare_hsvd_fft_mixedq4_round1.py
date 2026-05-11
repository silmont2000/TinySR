import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict


TRANSFORMER_A_SUFFIXES = [
    ".attn.to_q.base_layer",
    ".attn.to_k.base_layer",
    ".attn.to_v.base_layer",
    ".attn.to_out.0.base_layer",
    ".ff.net.0.proj.base_layer",
    ".ff.net.2.base_layer",
    ".proj_out",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Round1 quantization comparison: H-SVD vs FFT-MixedQ4 "
        "(A-level Transformer big Linear + B-level VAE pointwise only)."
    )
    parser.add_argument("--capture_dir", type=str, default="outputs/activations_capture_tinysr_env")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--budget_ratios", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--max_samples_per_call", type=int, default=1024)
    parser.add_argument("--max_samples_per_layer", type=int, default=4096)
    parser.add_argument("--max_layers", type=int, default=128)
    parser.add_argument("--bench_repeat", type=int, default=5)
    parser.add_argument("--hsvd_min_block", type=int, default=8)
    parser.add_argument("--hsvd_global_fracs", type=str, default="0.5")
    parser.add_argument("--hsvd_block_sizes", type=str, default="16,32,64")
    parser.add_argument("--skip_hsvd", action="store_true", help="Only run FFT-MixedQ4 when enabled.")
    parser.add_argument("--high_freq_bits", type=int, default=4)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--out_png", type=str, default="")
    return parser.parse_args()


def parse_budget_ratios(s: str):
    ratios = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        ratios.append(float(x))
    return ratios


def parse_int_list(s: str):
    values = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(int(x))
    return values


def parse_float_list(s: str):
    values = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(float(x))
    return values


def relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (a - b).norm() / (a.norm() + eps)


def quantize_symmetric(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    qmax = (2 ** (bits - 1)) - 1
    scale = x.abs().max().clamp_min(1e-12) / qmax
    return torch.clamp((x / scale).round(), -qmax, qmax) * scale


def fourier_mixedq4_with_budget(w: torch.Tensor, target_params: int, high_freq_bits: int = 4):
    # Budget is counted in "equivalent FP32 real params of original matrix".
    # This keeps budget ratios directly comparable to original weight size.
    f = torch.fft.fft2(w)
    mag = torch.abs(f)
    total = w.numel()
    fp_cost = 1.0
    q_cost = float(high_freq_bits) / 32.0
    min_budget = total * q_cost

    if target_params < min_budget:
        keep = max(1, min(total, int(target_params // fp_cost)))
        topk = torch.topk(mag.flatten(), keep).values
        threshold = topk[-1]
        mask = mag >= threshold
        f_hat = f * mask
        w_hat = torch.fft.ifft2(f_hat).real
        return w_hat, float(mask.sum().item()) * fp_cost

    keep_fp = int((target_params - total * q_cost) // max(fp_cost - q_cost, 1e-8))
    keep_fp = max(0, min(total, keep_fp))
    if keep_fp > 0:
        topk = torch.topk(mag.flatten(), keep_fp).values
        threshold = topk[-1]
        mask_fp = mag >= threshold
    else:
        mask_fp = torch.zeros_like(mag, dtype=torch.bool)

    f_high = f * (~mask_fp)
    f_high_q = quantize_symmetric(f_high.real, high_freq_bits) + 1j * quantize_symmetric(f_high.imag, high_freq_bits)
    f_hat = f * mask_fp + f_high_q
    w_hat = torch.fft.ifft2(f_hat).real

    actual_params = float(mask_fp.sum().item()) * fp_cost + float((~mask_fp).sum().item()) * q_cost
    return w_hat, actual_params


def truncated_svd_with_budget(w: torch.Tensor, target_params: int):
    m, n = w.shape
    rank = max(1, min(int(target_params // max(m + n, 1)), min(m, n)))
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    w_hat = u[:, :rank] @ torch.diag(s[:rank]) @ vh[:rank, :]
    return w_hat, rank * (m + n), rank


def hsvd_once(w: torch.Tensor, r_global: int, block_size: int):
    m, n = w.shape
    bh = block_size
    bw = block_size

    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    wg = u[:, :r_global] @ torch.diag(s[:r_global]) @ vh[:r_global, :]
    residual = w - wg

    nbm = m // bh
    nbn = n // bw
    if nbm == 0 or nbn == 0:
        return wg, r_global * (m + n)

    rr = residual[: nbm * bh, : nbn * bw]
    blocks = rr.view(nbm, bh, nbn, bw).permute(0, 2, 1, 3).reshape(-1, bh, bw)

    ub, sb, vhb = torch.linalg.svd(blocks, full_matrices=False)
    rank1 = (ub[:, :, :1] * sb[:, :1].unsqueeze(1)) @ vhb[:, :1, :]
    rec = rank1.reshape(nbm, nbn, bh, bw).permute(0, 2, 1, 3).reshape(nbm * bh, nbn * bw)

    w_hat = wg.clone()
    w_hat[: nbm * bh, : nbn * bw] = wg[: nbm * bh, : nbn * bw] + rec
    total_params = r_global * (m + n) + nbm * nbn * (bh + bw)
    return w_hat, total_params


def preprocess_image(image_path: str, process_size: int, upscale: int, device: torch.device, dtype: torch.dtype):
    image = Image.open(image_path).convert("RGB")
    ori_w, ori_h = image.size
    if ori_w < process_size // upscale or ori_h < process_size // upscale:
        scale = (process_size // upscale) / min(ori_w, ori_h)
        new_w, new_h = int(scale * ori_w), int(scale * ori_h)
    else:
        new_w, new_h = ori_w, ori_h
    new_w, new_h = new_w * upscale, new_h * upscale
    new_w -= new_w % 8
    new_h -= new_h % 8

    pixel_values = transforms.ToTensor()(image).unsqueeze(0)
    pixel_values = torch.nn.functional.interpolate(
        pixel_values, size=(new_h, new_w), mode="bicubic", align_corners=False
    )
    pixel_values = (pixel_values * 2 - 1).to(device, dtype=dtype)
    return pixel_values


def is_target_layer(name: str, module: nn.Module) -> bool:
    # Exclude all LoRA-injected modules from quantization.
    if "lora_" in name:
        return False

    # A-level: Transformer big Linear layers.
    if isinstance(module, nn.Linear) and name.startswith("transformer."):
        return any(name.endswith(suf) for suf in TRANSFORMER_A_SUFFIXES)

    # B-level: VAE pointwise 1x1 conv.
    if isinstance(module, nn.Conv2d) and name.startswith("vae.") and module.kernel_size == (1, 1) and module.groups == 1:
        return name.endswith(".pointwise")

    # C-level are intentionally excluded in round1.
    return False


def extract_input_matrix(module: nn.Module, x: torch.Tensor, max_rows: int):
    if isinstance(module, nn.Linear):
        mat = x.reshape(-1, x.shape[-1])
    elif isinstance(module, nn.Conv2d):
        mat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    else:
        return None

    rows = mat.shape[0]
    if rows > max_rows:
        idx = torch.linspace(0, rows - 1, steps=max_rows, device=mat.device).long()
        mat = mat.index_select(0, idx)
    return mat.detach().float().cpu()


def load_models_and_cfg(capture_dir: Path, device: torch.device, weight_dtype: torch.dtype):
    run_meta = json.loads((capture_dir / "run_meta.json").read_text(encoding="utf-8"))
    cfg = run_meta["args"]

    transformer = TinySD3Transformer2DModel.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )
    vae = AutoencoderTiny.from_pretrained(cfg["vae_path"], torch_dtype=weight_dtype)

    if cfg.get("lora_dir", ""):
        from peft import LoraConfig

        lora_rank = cfg.get("rank", 64)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(lora_cfg)
        transformer.enable_adapters()
        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(cfg["lora_dir"], weight_name="transformer.safetensors")
        load_lora_state_dict(lora_state_dict, transformer)

    transformer = transformer.to(device, dtype=weight_dtype).eval()
    vae = vae.to(device, dtype=weight_dtype).eval()
    return transformer, vae, cfg


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    capture_dir = Path(args.capture_dir)
    if not capture_dir.is_absolute():
        capture_dir = (repo_root / capture_dir).resolve()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    if device.type == "cpu":
        _orig_torch_load = torch.load

        def _patched_torch_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("map_location", "cpu")
            return _orig_torch_load(*load_args, **load_kwargs)

        torch.load = _patched_torch_load

    manifest = json.loads((capture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_layer_count = defaultdict(int)
    for item in manifest:
        manifest_layer_count[item["layer"]] += 1

    transformer, vae, cfg = load_models_and_cfg(capture_dir, device, weight_dtype)

    module_map = {}
    for n, m in vae.named_modules():
        module_map[f"vae.{n}" if n else "vae"] = m
    for n, m in transformer.named_modules():
        module_map[f"transformer.{n}" if n else "transformer"] = m

    candidates = []
    for name, module in module_map.items():
        if is_target_layer(name, module) and manifest_layer_count.get(name, 0) > 0 and hasattr(module, "weight"):
            candidates.append(
                {
                    "name": name,
                    "module": module,
                    "weight_params": int(module.weight.numel()),
                    "manifest_calls": int(manifest_layer_count[name]),
                }
            )
    candidates.sort(key=lambda x: x["weight_params"], reverse=True)
    candidates = candidates[: args.max_layers]
    print(f"Selected target layers: {len(candidates)}")

    layer_samples = defaultdict(list)
    sampled_rows = defaultdict(int)
    hooks = []
    for item in candidates:
        layer_name = item["name"]
        module = item["module"]

        def make_hook(name=layer_name, mod=module):
            def hook_fn(_module, inputs):
                if len(inputs) == 0:
                    return
                if sampled_rows[name] >= args.max_samples_per_layer:
                    return
                mat = extract_input_matrix(mod, inputs[0], args.max_samples_per_call)
                if mat is None or mat.numel() == 0:
                    return
                remain = args.max_samples_per_layer - sampled_rows[name]
                mat = mat[:remain]
                layer_samples[name].append(mat)
                sampled_rows[name] += mat.shape[0]

            return hook_fn

        hooks.append(module.register_forward_pre_hook(make_hook()))

    pooled_prompt_embeds = torch.load(os.path.join(cfg["embedding_dir"], "pool_embeds.pt"), map_location=device).to(
        dtype=weight_dtype
    )
    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)
    pixel_values = preprocess_image(cfg["input_image"], cfg["process_size"], cfg["upscale"], device, weight_dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        model_pred = transformer(
            hidden_states=model_input,
            timestep=timesteps,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        latent_stu = model_input - model_pred
        _ = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0]

    for h in hooks:
        h.remove()

    budget_ratios = parse_budget_ratios(args.budget_ratios)
    hsvd_global_fracs = parse_float_list(args.hsvd_global_fracs)
    hsvd_block_sizes = parse_int_list(args.hsvd_block_sizes)
    if len(hsvd_global_fracs) == 0:
        hsvd_global_fracs = [0.5]
    if len(hsvd_block_sizes) == 0:
        hsvd_block_sizes = [16, 32, 64]
    rows = []
    methods = ["fft_mixedq4"] if args.skip_hsvd else ["hsvd", "fft_mixedq4"]
    summary = {r: {m: {"w_sum": 0.0, "werr": 0.0, "yerr": 0.0, "lat": 0.0, "pr": 0.0} for m in methods} for r in budget_ratios}

    def bench_layer(x, w, bias):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.bench_repeat):
                y = x @ w.t()
                if bias is not None:
                    y = y + bias
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / args.bench_repeat

    total_layers = len(candidates)
    for layer_idx, item in enumerate(candidates, start=1):
        name = item["name"]
        module = item["module"]
        if len(layer_samples[name]) == 0:
            continue
        print(f"[Layer {layer_idx}/{total_layers}] {name} - start")

        x = torch.cat(layer_samples[name], dim=0).to(device=device, dtype=torch.float32)
        if isinstance(module, nn.Linear):
            w = module.weight.detach().to(device=device, dtype=torch.float32)
            bias = module.bias.detach().to(device=device, dtype=torch.float32) if module.bias is not None else None
        elif isinstance(module, nn.Conv2d):
            w = module.weight[:, :, 0, 0].detach().to(device=device, dtype=torch.float32)
            bias = module.bias.detach().to(device=device, dtype=torch.float32) if module.bias is not None else None
        else:
            continue

        orig_params = w.numel()
        y_orig = x @ w.t()
        if bias is not None:
            y_orig = y_orig + bias

        for br in budget_ratios:
            target = max(1, int(orig_params * br))

            if not args.skip_hsvd:
                # H-SVD: compact config search around budget.
                best = None
                max_rank = min(w.shape)
                for gf in hsvd_global_fracs:
                    rg = max(1, min(int((target * gf) // max(w.shape[0] + w.shape[1], 1)), max_rank))
                    for bs in sorted(set([args.hsvd_min_block] + hsvd_block_sizes)):
                        if bs > min(w.shape):
                            continue
                        try:
                            w_h, p_h = hsvd_once(w, rg, bs)
                        except RuntimeError:
                            continue
                        if p_h > target:
                            continue
                        y_h = x @ w_h.t()
                        if bias is not None:
                            y_h = y_h + bias
                        yerr_h = float(relative_error(y_orig, y_h).item())
                        if best is None or yerr_h < best["output_rel_err"]:
                            best = {
                                "w_hat": w_h,
                                "param_count": int(p_h),
                                "output_rel_err": yerr_h,
                                "r_global": int(rg),
                                "block_size": int(bs),
                            }

                if best is not None:
                    w_h = best["w_hat"]
                    y_h = x @ w_h.t()
                    if bias is not None:
                        y_h = y_h + bias
                    werr_h = float(relative_error(w, w_h).item())
                    yerr_h = float(relative_error(y_orig, y_h).item())
                    lat_h = float(bench_layer(x, w_h, bias))

                    rows.append(
                        {
                            "layer": name,
                            "method": "hsvd",
                            "budget_ratio": br,
                            "target_params": int(target),
                            "actual_params": int(best["param_count"]),
                            "actual_param_ratio": float(best["param_count"]) / float(orig_params),
                            "weight_rel_err": werr_h,
                            "output_rel_err": yerr_h,
                            "latency_ms": lat_h,
                            "hsvd_r_global": best["r_global"],
                            "hsvd_block_size": best["block_size"],
                            "orig_params": int(orig_params),
                        }
                    )
                    s = summary[br]["hsvd"]
                    wgt = float(orig_params)
                    s["w_sum"] += wgt
                    s["werr"] += werr_h * wgt
                    s["yerr"] += yerr_h * wgt
                    s["lat"] += lat_h * wgt
                    s["pr"] += (float(best["param_count"]) / float(orig_params)) * wgt

            # FFT-MixedQ4
            w_f, p_f = fourier_mixedq4_with_budget(w, target, high_freq_bits=args.high_freq_bits)
            y_f = x @ w_f.t()
            if bias is not None:
                y_f = y_f + bias
            werr_f = float(relative_error(w, w_f).item())
            yerr_f = float(relative_error(y_orig, y_f).item())
            lat_f = float(bench_layer(x, w_f, bias))
            rows.append(
                {
                    "layer": name,
                    "method": "fft_mixedq4",
                    "budget_ratio": br,
                    "target_params": int(target),
                    "actual_params": float(p_f),
                    "actual_param_ratio": float(p_f) / float(orig_params),
                    "weight_rel_err": werr_f,
                    "output_rel_err": yerr_f,
                    "latency_ms": lat_f,
                    "orig_params": int(orig_params),
                }
            )
            s = summary[br]["fft_mixedq4"]
            wgt = float(orig_params)
            s["w_sum"] += wgt
            s["werr"] += werr_f * wgt
            s["yerr"] += yerr_f * wgt
            s["lat"] += lat_f * wgt
            s["pr"] += (float(p_f) / float(orig_params)) * wgt
        print(f"[Layer {layer_idx}/{total_layers}] {name} - done")

    summary_rows = []
    for br in budget_ratios:
        for method in methods:
            s = summary[br][method]
            if s["w_sum"] <= 0:
                continue
            summary_rows.append(
                {
                    "budget_ratio": br,
                    "method": method,
                    "weighted_weight_rel_err": s["werr"] / s["w_sum"],
                    "weighted_output_rel_err": s["yerr"] / s["w_sum"],
                    "weighted_latency_ms": s["lat"] / s["w_sum"],
                    "weighted_actual_param_ratio": s["pr"] / s["w_sum"],
                }
            )

    out_json = Path(args.out_json) if args.out_json else (capture_dir / "round1_hsvd_fft_mixedq4.json")
    out_png = Path(args.out_png) if args.out_png else (capture_dir / "round1_hsvd_fft_mixedq4.png")
    out_csv = out_json.with_suffix(".csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "selected_layers": [x["name"] for x in candidates],
                "rows": rows,
                "summary": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "budget_ratio",
                "method",
                "weighted_weight_rel_err",
                "weighted_output_rel_err",
                "weighted_latency_ms",
                "weighted_actual_param_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method in methods:
        xs = [x["budget_ratio"] for x in summary_rows if x["method"] == method]
        y_err = [x["weighted_output_rel_err"] for x in summary_rows if x["method"] == method]
        y_param = [x["weighted_actual_param_ratio"] for x in summary_rows if x["method"] == method]
        y_lat = [x["weighted_latency_ms"] for x in summary_rows if x["method"] == method]
        axes[0].plot(xs, y_err, marker="o", label=method)
        axes[1].plot(xs, y_param, marker="o", label=method)
        axes[2].plot(xs, y_lat, marker="o", label=method)

    axes[0].set_title("Weighted Output Relative Error")
    axes[1].set_title("Weighted Actual Param Ratio")
    axes[2].set_title("Weighted Layer Matmul Latency (ms)")
    for ax in axes:
        ax.set_xlabel("Target Budget Ratio")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)

    print(f"[DONE] rows={len(rows)}")
    print(f"[DONE] summary={len(summary_rows)}")
    print(f"[DONE] json={out_json}")
    print(f"[DONE] csv={out_csv}")
    print(f"[DONE] png={out_png}")


if __name__ == "__main__":
    main()
