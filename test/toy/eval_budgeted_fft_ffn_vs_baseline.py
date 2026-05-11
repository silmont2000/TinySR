import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline

sys.path.append(".")

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.mixed_linear_fftq4 import MixedLinearFFTQ4
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import bitsandbytes as bnb

A_FFN_SUFFIXES = [
    ".ff.net.0.proj.base_layer",
    ".ff.net.2.base_layer",
    ".ff.net.0.proj",
    ".ff.net.2",
]

A_FULL_EXTRA_SUFFIXES = [
    ".attn.to_q.base_layer",
    ".attn.to_k.base_layer",
    ".attn.to_v.base_layer",
    ".attn.to_out.0.base_layer",
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    ".attn.to_out.0",
    ".proj_out.base_layer",
    ".proj_out",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Budgeted FFT-MixedQ4 for FFN layers and full-image baseline comparison.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--input_dir", type=str, default="dataset/test_image/")
    parser.add_argument("--gt_dir", type=str, default="", help="Optional GT folder with same image names.")
    parser.add_argument("--output_root", type=str, default="outputs/fft_ffn_budget_eval")

    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="adain")

    parser.add_argument("--budget_ratios", type=str, default="0.15,0.2,0.25")
    parser.add_argument("--high_freq_bits", type=int, default=4)
    parser.add_argument("--quant_scope", type=str, choices=["ffn_only", "a_full"], default="ffn_only")
    parser.add_argument(
        "--compute_impl",
        type=str,
        choices=["mixed_linear", "dense_reconstruct", "weight_only_4bit", "weight_only_8bit"],
        default="mixed_linear",
        help="mixed_linear: low-freq FP + high-freq INT4 branch; dense_reconstruct: overwrite dense weight only.",
    )
    parser.add_argument("--warmup_images", type=int, default=1, help="Warmup runs before timed inference.")
    return parser.parse_args()


def parse_budget_ratios(s: str):
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


# 用来给每个 tile 生成一个 中心大、边缘小的二维高斯融合权重图，从而在重叠拼接时减少 tile 接缝和边缘伪影
def _gaussian_weights(tile_width, tile_height, nbatches, in_channels, device):
    from numpy import pi, exp, sqrt

    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    midpoint_y = tile_height / 2
    x_probs = [exp(-(x - midpoint_x) * (x - midpoint_x) / (tile_width * tile_width) / (2 * var)) / sqrt(2 * pi * var) for x in range(tile_width)]
    y_probs = [exp(-(y - midpoint_y) * (y - midpoint_y) / (tile_height * tile_height) / (2 * var)) / sqrt(2 * pi * var) for y in range(tile_height)]
    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device), (nbatches, in_channels, 1, 1))


def tile_sample(lq_latent, transformer, timesteps, pooled_prompt_embeds, weight_dtype, latent_tiled_size=64, latent_tiled_overlap=8):
    with torch.no_grad():
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (latent_tiled_size, latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred = transformer(
                hidden_states=lq_latent,
                timestep=timesteps,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        else:
            tile_size = min(tile_size, min(h, w))
            tile_weights = _gaussian_weights(tile_size, tile_size, 1, transformer.config.in_channels, lq_latent.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    else:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size
                    else:
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)

                    input_tile = lq_latent[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size]
                    pred = transformer(
                        hidden_states=input_tile.to(lq_latent.device, dtype=weight_dtype),
                        timestep=timesteps,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    noise_preds.append(pred)

            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    else:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size
                    else:
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)

                    noise_pred[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += noise_preds[row * grid_cols + col] * tile_weights
                    contributors[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += tile_weights
            model_pred = noise_pred / contributors.clamp_min(1e-8)
    return model_pred.to(lq_latent.device, dtype=weight_dtype)


def quantize_symmetric(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    qmax = (2 ** (bits - 1)) - 1
    scale = x.abs().max().clamp_min(1e-12) / qmax
    return torch.clamp((x / scale).round(), -qmax, qmax) * scale


def fourier_mixedq4_with_budget(w: torch.Tensor, target_params: int, high_freq_bits: int = 4):
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
    # f_hat = f * mask_fp
    # w_hat = torch.fft.ifft2(f_hat).real
    actual_params = float(mask_fp.sum().item()) * fp_cost + float((~mask_fp).sum().item()) * q_cost
    return w_hat, actual_params


def apply_fft_budget_to_a_layers(transformer: nn.Module, budget_ratio: float, high_freq_bits: int, quant_scope: str):
    if quant_scope == "a_full":
        target_suffixes = A_FFN_SUFFIXES + A_FULL_EXTRA_SUFFIXES
    else:
        target_suffixes = A_FFN_SUFFIXES
    touched = []
    for name, module in transformer.named_modules():
        full_name = f"transformer.{name}" if name else "transformer"
        if "lora_" in full_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suf) for suf in target_suffixes):
            continue
        with torch.no_grad():
            if quant_scope not in ["ffn_only", "a_full"]:
                continue
            # This function keeps backward compatibility but now delegates to branch replacement logic.
            # Actual replacement mode is decided by caller via `compute_impl`.
            w = module.weight.detach().float()
            target = max(1, int(w.numel() * budget_ratio))
            w_hat, actual_params = fourier_mixedq4_with_budget(w, target, high_freq_bits=high_freq_bits)
            module.weight.copy_(w_hat.to(module.weight.dtype))
            touched.append(
                {
                    "layer": full_name,
                    "impl": "dense_reconstruct",
                    "orig_params": int(w.numel()),
                    "target_params": int(target),
                    "actual_params": float(actual_params),
                    "actual_ratio": float(actual_params) / float(w.numel()),
                }
            )
    return touched


def replace_a_layers_with_weight_only(transformer: nn.Module, quant_scope: str, wbits: int, compute_dtype: torch.dtype):
    if quant_scope == "a_full":
        target_suffixes = A_FFN_SUFFIXES + A_FULL_EXTRA_SUFFIXES
    else:
        target_suffixes = A_FFN_SUFFIXES

    touched = []
    for name, module in list(transformer.named_modules()):
        full_name = f"transformer.{name}" if name else "transformer"
        if "lora_" in full_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suf) for suf in target_suffixes):
            continue

        # 构造新层
        if wbits == 4:
            new_mod = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None),
                compute_dtype=compute_dtype,   # 一般 fp16
                quant_type="nf4",              # 可换 "fp4"
                compress_statistics=True,
            )
        elif wbits == 8:
            new_mod = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None),
                has_fp16_weights=False,
            )
        else:
            raise ValueError(f"Unsupported wbits={wbits}")

        # 拷权重
        with torch.no_grad():
            new_mod.weight.copy_(module.weight.detach())
            if module.bias is not None:
                new_mod.bias.copy_(module.bias.detach())

        new_mod = new_mod.to(module.weight.device)

        # 替换到父模块
        parent_name, child_name = (name.rsplit(".", 1) if "." in name else ("", name))
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(parent, child_name, new_mod)
        print(
            f"[Q-REPLACE] {full_name}: "
            f"{type(module).__name__} -> {type(new_mod).__name__}, "
            f"wbits={wbits}, in={module.in_features}, out={module.out_features}, "
            f"bias={module.bias is not None}, dev={module.weight.device}"
        )
        touched.append({
            "layer": full_name,
            "impl": f"weight_only_{wbits}bit",
            "backend": "bitsandbytes",
        })
    print(f"[Q-REPLACE] total replaced = {len(touched)}")
    return touched

def replace_a_layers_with_mixed_linear(transformer: nn.Module, budget_ratio: float, high_freq_bits: int, quant_scope: str):
    if quant_scope == "a_full":
        target_suffixes = A_FFN_SUFFIXES + A_FULL_EXTRA_SUFFIXES
    else:
        target_suffixes = A_FFN_SUFFIXES

    touched = []
    for name, module in list(transformer.named_modules()):
        full_name = f"transformer.{name}" if name else "transformer"
        if "lora_" in full_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suf) for suf in target_suffixes):
            continue

        mixed = MixedLinearFFTQ4.from_linear(
            module,
            budget_ratio=budget_ratio,
            high_freq_bits=high_freq_bits,
        )
        mixed = mixed.to(module.weight.device)
        parent_name, child_name = (name.rsplit(".", 1) if "." in name else ("", name))
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(parent, child_name, mixed)

        # 记录详细元信息
        touched.append({
            "layer": full_name,
            "impl": "mixed_linear",
            "high_freq_bits": high_freq_bits,
            "spectral_budget_ratio": mixed.meta.get("spectral_budget_ratio", 0),
            "impl_byte_ratio": mixed.meta.get("impl_byte_ratio", 0),
            "backend": mixed.meta.get("backend", "unknown")
        })
    return touched


def load_models(args, device, weight_dtype):
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    )
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)

    if args.lora_dir:
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            args.lora_dir, weight_name="transformer.safetensors", cache_dir=args.cache_dir
        )
        load_lora_state_dict(transformer_lora_state_dict, transformer)

    vae = vae.to(device, dtype=weight_dtype).eval()
    transformer = transformer.to(device, dtype=weight_dtype).eval()
    return transformer, vae


def preprocess_one_image(lr, upscale, process_size):
    ori_width, ori_height = lr.size
    resize_flag = False
    if ori_width < process_size // upscale or ori_height < process_size // upscale:
        scale = (process_size // upscale) / min(ori_width, ori_height)
        new_width, new_height = int(scale * ori_width), int(scale * ori_height)
        resize_flag = True
    else:
        new_width, new_height = ori_width, ori_height
    new_width, new_height = upscale * new_width, upscale * new_height
    if new_width % 8 or new_height % 8:
        resize_flag = True
        new_width = new_width - new_width % 8
        new_height = new_height - new_height % 8
    return resize_flag, new_width, new_height, ori_width, ori_height


def run_inference_set(args, mode_name, budget_ratio=None):
    device = torch.device(args.device)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    transformer, vae = load_models(args, device, weight_dtype)

    quant_meta = []
    if budget_ratio is not None:
        if args.compute_impl == "mixed_linear":
            quant_meta = replace_a_layers_with_mixed_linear(
                transformer, budget_ratio, args.high_freq_bits, args.quant_scope
            )
        elif args.compute_impl == "dense_reconstruct":
            quant_meta = apply_fft_budget_to_a_layers(
                transformer, budget_ratio, args.high_freq_bits, args.quant_scope
            )
        elif args.compute_impl == "weight_only_4bit":
            quant_meta = replace_a_layers_with_weight_only(
                transformer, args.quant_scope, wbits=4, compute_dtype=weight_dtype
            )
            print(f"[Q-META] impl={args.compute_impl}, scope={args.quant_scope}, layers={len(quant_meta)}")
            if len(quant_meta) > 0:
                print("[Q-META first 3]", quant_meta[:3])
        elif args.compute_impl == "weight_only_8bit":
            quant_meta = replace_a_layers_with_weight_only(
                transformer, args.quant_scope, wbits=8, compute_dtype=weight_dtype
            )
    pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=device).to(dtype=weight_dtype)
    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)

    if os.path.isdir(args.input_dir):
        image_names = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    else:
        image_names = [args.input_dir]
    os.makedirs(args.output_root, exist_ok=True)
    out_dir = os.path.join(args.output_root, mode_name)
    os.makedirs(out_dir, exist_ok=True)

    total_time = 0.0
    per_image_time = []
    tensor_transforms = transforms.Compose([transforms.ToTensor()])
    warmup_left = max(0, args.warmup_images)
    with torch.no_grad():
        for image_name in tqdm(image_names, desc=f"Infer[{mode_name}]"):
            lr = Image.open(image_name).convert("RGB")
            resize_flag, new_width, new_height, ori_width, ori_height = preprocess_one_image(
                lr, args.upscale, args.process_size
            )
            lr_scale = lr.resize((int(ori_width * args.upscale), int(ori_height * args.upscale)))
            pixel_values = tensor_transforms(lr).unsqueeze(0).to(device=device, dtype=weight_dtype)

            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(new_height, new_width), mode="bicubic", align_corners=False
            )
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to(device=device, dtype=weight_dtype)

            start_time = time.time()
            model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
            model_pred = tile_sample(model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype)
            latent_stu = model_input - model_pred
            image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            if warmup_left > 0:
                warmup_left -= 1
            else:
                total_time += elapsed
                per_image_time.append(elapsed)

            image_pil = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
            if resize_flag:
                image_pil = image_pil.resize((int(ori_width * args.upscale), int(ori_height * args.upscale)))
            if args.align_method == "adain":
                image_pil = adain_color_fix(target=image_pil, source=lr)
            elif args.align_method == "wavelet":
                image_pil = wavelet_color_fix(target=image_pil, source=lr_scale)

            image_pil.save(os.path.join(out_dir, os.path.basename(image_name)))
            if device.type == "cuda":
                torch.cuda.empty_cache()
            


            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
            ) as prof:
                with record_function("one_forward"):
                    _ = transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

            # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))

    if len(per_image_time) > 0:
        avg_time = float(np.mean(per_image_time))
        p50_time = float(np.percentile(per_image_time, 50))
        p90_time = float(np.percentile(per_image_time, 90))
    else:
        avg_time = None
        p50_time = None
        p90_time = None
    return {
        "mode": mode_name,
        "output_dir": out_dir,
        "image_count": len(image_names),
        "avg_time_sec": avg_time,
        "p50_time_sec": p50_time,
        "p90_time_sec": p90_time,
        "timed_image_count": len(per_image_time),
        "warmup_images": int(args.warmup_images),
        "compute_impl": args.compute_impl,
        "quant_layers": quant_meta,
    }


def compute_psnr(a: np.ndarray, b: np.ndarray):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10((255.0 * 255.0) / mse)


def compute_ssim_gray(a: np.ndarray, b: np.ndarray):
    # a, b: uint8 gray
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def compute_ssim_rgb(a: np.ndarray, b: np.ndarray):
    return float(np.mean([compute_ssim_gray(a[:, :, i], b[:, :, i]) for i in range(3)]))


def evaluate_dirs(ref_dir, test_dir):
    ref_imgs = sorted(glob.glob(os.path.join(ref_dir, "*.png")))
    scores = []
    for ref_path in ref_imgs:
        name = os.path.basename(ref_path)
        test_path = os.path.join(test_dir, name)
        if not os.path.exists(test_path):
            continue
        ref = np.array(Image.open(ref_path).convert("RGB"))
        test = np.array(Image.open(test_path).convert("RGB"))
        if ref.shape != test.shape:
            continue
        scores.append(
            {
                "name": name,
                "psnr": compute_psnr(ref, test),
                "ssim": compute_ssim_rgb(ref, test),
            }
        )
    if len(scores) == 0:
        return {"count": 0, "psnr": None, "ssim": None, "per_image": []}
    return {
        "count": len(scores),
        "psnr": float(np.mean([x["psnr"] for x in scores])),
        "ssim": float(np.mean([x["ssim"] for x in scores])),
        "per_image": scores,
    }


def main():
    args = parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    budgets = parse_budget_ratios(args.budget_ratios)
    os.makedirs(args.output_root, exist_ok=True)

    # # 1) baseline inference
    # baseline = run_inference_set(args, mode_name="baseline", budget_ratio=None)
    # baseline_dir = baseline["output_dir"]
    # print(
    #     "[BASELINE] "
    #     f"avg={baseline['avg_time_sec']:.4f}s p50={baseline['p50_time_sec']:.4f}s p90={baseline['p90_time_sec']:.4f}s "
    #     f"out={baseline_dir}"
    # )

    # # baseline vs GT if provided
    # baseline_gt = None
    # if args.gt_dir and os.path.isdir(args.gt_dir):
    #     baseline_gt = evaluate_dirs(args.gt_dir, baseline_dir)
    #     print(
    #         "[BASELINE vs GT] "
    #         f"count={baseline_gt['count']} psnr={baseline_gt['psnr']:.4f} ssim={baseline_gt['ssim']:.6f}"
    #     )

    all_rows = []
    all_results = []

    # 2) quantized inference for each budget
    for br in budgets:
        tag = f"fft_ffn_b{int(round(br * 100)):02d}"
        run_res = run_inference_set(args, mode_name=tag, budget_ratio=br)
        out_dir = run_res["output_dir"]
        # vs_baseline = evaluate_dirs(baseline_dir, out_dir)
        # print(
        #     f"[{tag}] avg={run_res['avg_time_sec']:.4f}s p50={run_res['p50_time_sec']:.4f}s p90={run_res['p90_time_sec']:.4f}s "
        #     f"vs_baseline psnr={vs_baseline['psnr']:.4f} ssim={vs_baseline['ssim']:.6f}"
        # )

        vs_gt = None
        if args.gt_dir and os.path.isdir(args.gt_dir):
            vs_gt = evaluate_dirs(args.gt_dir, out_dir)
            print(
                f"[{tag} vs GT] count={vs_gt['count']} "
                f"psnr={vs_gt['psnr']:.4f} ssim={vs_gt['ssim']:.6f}"
            )

        # all_results.append(
        #     {
        #         "budget_ratio": br,
        #         "run": run_res,
        #         "vs_baseline": vs_baseline,
        #         "vs_gt": vs_gt,
        #     }
        # )
        # all_rows.append(
        #     {
        #         "budget_ratio": br,
        #         "baseline_avg_time_sec": baseline["avg_time_sec"],
        #         "baseline_p50_time_sec": baseline["p50_time_sec"],
        #         "baseline_p90_time_sec": baseline["p90_time_sec"],
        #         "quant_avg_time_sec": run_res["avg_time_sec"],
        #         "quant_p50_time_sec": run_res["p50_time_sec"],
        #         "quant_p90_time_sec": run_res["p90_time_sec"],
        #         "speedup_vs_baseline_avg": baseline["avg_time_sec"] / max(run_res["avg_time_sec"], 1e-8),
        #         "speedup_vs_baseline_p50": baseline["p50_time_sec"] / max(run_res["p50_time_sec"], 1e-8),
        #         "speedup_vs_baseline_p90": baseline["p90_time_sec"] / max(run_res["p90_time_sec"], 1e-8),
        #         "psnr_vs_baseline": vs_baseline["psnr"],
        #         "ssim_vs_baseline": vs_baseline["ssim"],
        #         "psnr_vs_gt": None if vs_gt is None else vs_gt["psnr"],
        #         "ssim_vs_gt": None if vs_gt is None else vs_gt["ssim"],
        #     }
        # )

    report = {
        "args": vars(args),
        # "baseline": baseline,
        # "baseline_vs_gt": baseline_gt,
        "budgets": all_results,
    }
    json_path = os.path.join(args.output_root, "budget_fft_ffn_eval_report.json")
    csv_path = os.path.join(args.output_root, "budget_fft_ffn_eval_report.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "budget_ratio",
                "baseline_avg_time_sec",
                "baseline_p50_time_sec",
                "baseline_p90_time_sec",
                "quant_avg_time_sec",
                "quant_p50_time_sec",
                "quant_p90_time_sec",
                "speedup_vs_baseline_avg",
                "speedup_vs_baseline_p50",
                "speedup_vs_baseline_p90",
                "psnr_vs_baseline",
                "ssim_vs_baseline",
                "psnr_vs_gt",
                "ssim_vs_gt",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[DONE] report json: {json_path}")
    print(f"[DONE] report csv : {csv_path}")


if __name__ == "__main__":
    main()
