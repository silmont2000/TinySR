#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fmt: off
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.tiler import gaussian_weights, tile_sample
from utils.util import load_lora_state_dict
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm
from torchvision import transforms
from peft import LoraConfig
from PIL import Image
import torch
import numpy as np
from collections import defaultdict
import os
import math
import json
import glob
import argparse
# fmt: on

"""
Activation pre‑analysis for TinySR:
...
"""
sys.path.append(".")


# ★ 必须在所有本地 import 之前
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------
# 工具函数
# -------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_images(root: str, max_num: int = 200) -> list:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(root, e)))
    imgs = sorted(imgs)
    if max_num > 0:
        imgs = imgs[:max_num]
    return imgs


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# 激活值收集与分析
# -------------------------------

class ActivationCollector:
    """按数据集分开存储每层激活值，支持后续计算各类指标"""

    def __init__(self, max_samples_per_layer=500, max_qq_points=20000, seed=42):
        self.data = defaultdict(lambda: {"A": [], "B": []})
        self.max_samples = max_samples_per_layer
        self.max_qq_points = max_qq_points
        self.rng = np.random.default_rng(seed)

    def add(self, layer_name: str, dataset_tag: str, x: torch.Tensor):
        # x shape can be arbitrary, we flatten to 1D for storage
        if dataset_tag not in ("A", "B"):
            raise ValueError("dataset_tag must be 'A' or 'B'")
        # Detach and move to CPU
        arr = x.detach().float().cpu().numpy().ravel()
        # Subsampling to avoid memory explosion
        if arr.size > self.max_qq_points:
            idx = self.rng.choice(
                arr.size, size=self.max_qq_points, replace=False)
            arr = arr[idx]
        # Limit total stored vectors per layer per dataset
        store_list = self.data[layer_name][dataset_tag]
        if len(store_list) < self.max_samples:
            store_list.append(arr)


def compute_layer_stats(vectors_list):
    """
    Given a list of 1D numpy arrays (all from one dataset for one layer),
    compute:
      - global mean, std
      - excess kurtosis (using all data concatenated)
      - outlier ratio based on global mean ± 5*std
      - channel anomaly frequency variance (only if input was [B, C, H, W] originally,
        but here we only have 1D flattened; we approximate by reshaping if possible.
        Since we stored flattened 1D array, channel info is lost.
        To fix this, we should save per-channel statistics during collection.
        Let's modify the collector to also save per‑channel mean abs for this purpose.
        We'll implement a separate channel anomaly collector.
    """
    if not vectors_list:
        return None

    all_vals = np.concatenate(vectors_list)
    n = all_vals.size
    mu = np.mean(all_vals)
    std = np.std(all_vals)
    # kurtosis (excess)
    if std < 1e-12:
        kurt = 0.0
    else:
        kurt = np.mean((all_vals - mu) ** 4) / (std ** 4) - 3.0

    # outlier ratio: > mu + 5*std or < mu - 5*std
    threshold = 5.0 * std
    outlier_mask = np.abs(all_vals - mu) > threshold
    outlier_ratio = np.mean(outlier_mask)

    # For per‑sample kurtosis (to see stability)
    sample_kurts = []
    for v in vectors_list:
        if v.size < 4:
            continue
        s_mu = v.mean()
        s_std = v.std()
        if s_std < 1e-12:
            sample_kurts.append(0.0)
        else:
            sample_kurts.append(np.mean((v - s_mu) ** 4) / (s_std ** 4) - 3.0)
    avg_sample_kurt = np.mean(sample_kurts) if sample_kurts else 0.0

    return {
        "n_total": int(n),
        "global_mean": float(mu),
        "global_std": float(std),
        "kurtosis_excess": float(kurt),
        "outlier_ratio_global": float(outlier_ratio),
        "avg_per_sample_kurtosis": float(avg_sample_kurt),
    }


# 通道异常频率需要保留通道维度，所以单独设计一个收集器
class ChannelAnomalyCollector:
    def __init__(self, topk_ratio=0.01):
        # list of per-sample anomaly channel indices
        self.per_layer = defaultdict(lambda: {"A": [], "B": []})
        self.topk_ratio = topk_ratio

    def add(self, layer_name: str, dataset_tag: str, x: torch.Tensor):
        # x shape: (batch, C, ...)  or (batch, tokens, C)
        if x.dim() < 2:
            return
        x_abs = x.detach().float().abs()
        # Assume last dimension is channel
        *_, C = x.shape
        # Flatten all but channel
        x_reshape = x_abs.reshape(-1, C)  # (T, C)
        ch_mean = x_reshape.mean(dim=0)  # (C,)
        k = max(1, int(math.ceil(C * self.topk_ratio)))
        topk_idx = torch.topk(ch_mean, k=k).indices.cpu().numpy().tolist()
        self.per_layer[layer_name][dataset_tag].append(topk_idx)

    def compute(self, layer_name):
        data = self.per_layer.get(layer_name)
        if not data:
            return None
        setA = data["A"]
        setB = data["B"]
        # 计算频率方差：先合并两个数据集？我们分别报告，并在策略中使用

        def _calc_freq_var(samples):
            if not samples:
                return None, None, None
            # 计算每个通道被选为异常的频率
            # 需要知道总通道数，从任意一个样本的索引最大值推断
            all_indices = [idx for sample in samples for idx in sample]
            if not all_indices:
                return None, None, None
            max_ch = max(all_indices)
            freq = np.zeros(max_ch + 1, dtype=np.float64)
            for sample in samples:
                for ch in sample:
                    freq[ch] += 1
            freq /= len(samples)
            mean_freq = float(freq.mean())
            var_freq = float(freq.var())
            # top-static channels
            top5_idx = np.argsort(-freq)[:5]
            top_static = [
                {"channel": int(i), "freq": float(freq[i])} for i in top5_idx]
            return mean_freq, var_freq, top_static

        mean_a, var_a, top_a = _calc_freq_var(setA)
        mean_b, var_b, top_b = _calc_freq_var(setB)

        # 综合两个数据集的平均方差作为指标
        avg_var = np.nanmean([var_a, var_b]) if (
            var_a is not None and var_b is not None) else None
        return {
            "setA": {"mean_freq": mean_a, "var_freq": var_a, "top_static_channels": top_a},
            "setB": {"mean_freq": mean_b, "var_freq": var_b, "top_static_channels": top_b},
            "avg_var": avg_var
        }


# -------------------------------
# 主分析脚本
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Activation pre-analysis for TinySR")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str,
                        default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str,
                        default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/activation_preanalysis")

    parser.add_argument("--dataset_A", type=str,
                        default="/data/disk2/xby/TinySR/dataset/StableSR_testsets/RealSRVal_crop128/test_LR")
    parser.add_argument("--dataset_B", type=str,
                        default="/data/disk2/xby/TinySR/dataset/StableSR_testsets/DrealSRVal_crop128/test_LR")

    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str,
                        choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--align_method", type=str,
                        choices=["wavelet", "adain", "nofix"], default="adain")

    parser.add_argument("--max_images_per_set", type=int, default=50,
                        help="限制每个数据集使用的图片数量（0 表示全部）")
    parser.add_argument("--max_qq_points", type=int, default=20000,
                        help="每层每次采样最多保留多少激活值用于 Q-Q/KS 分析")
    parser.add_argument("--max_ks_pairs_per_layer", type=int, default=32,
                        help="每层最多采样多少组样本对用于 KS 统计；当前版本预留参数")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_weight_dtype(mixed_precision):
    return torch.float16 if mixed_precision == "fp16" else torch.float32


def load_models(args, device, weight_dtype):
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    )
    vae = AutoencoderTiny.from_pretrained(
        args.vae_path,
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    )
    if args.lora_dir:
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0",
                            "proj", "linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            args.lora_dir, weight_name="transformer.safetensors", cache_dir=args.cache_dir)
        load_lora_state_dict(transformer_lora_state_dict, transformer)

    transformer = transformer.to(device, dtype=weight_dtype).eval()
    vae = vae.to(device, dtype=weight_dtype).eval()
    return transformer, vae


# -------------------------------
# Hook 注册
# -------------------------------

def register_hooks(transformer, collector_val, collector_ch, current_dataset_tag):
    """
    对 transformer 中所有 Linear 层的输入注册 hook，
    同时收集展平后的激活值（用于分布统计）和通道级信息（用于异常通道分析）。
    current_dataset_tag: 'A' 或 'B'
    """
    handles = []
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            def make_hook(layer_name, tag):
                def hook(module, input):
                    if not input:
                        return
                    x = input[0]
                    if isinstance(x, torch.Tensor):
                        collector_val.add(layer_name, tag, x)
                        collector_ch.add(layer_name, tag, x)
                return hook
            h = module.register_forward_pre_hook(
                make_hook(name, current_dataset_tag))
            handles.append(h)
    return handles




def preprocess_image(lr, upscale, process_size):
    ori_width, ori_height = lr.size
    resize_flag = False
    if ori_width < process_size // upscale or ori_height < process_size // upscale:
        scale = (process_size // upscale) / min(ori_width, ori_height)
        new_width = int(scale * ori_width)
        new_height = int(scale * ori_height)
        resize_flag = True
    else:
        new_width = ori_width
        new_height = ori_height
    new_width = upscale * new_width
    new_height = upscale * new_height
    if new_width % 8 or new_height % 8:
        resize_flag = True
        new_width = new_width - new_width % 8
        new_height = new_height - new_height % 8
    return resize_flag, new_width, new_height, ori_width, ori_height


def image_to_latent(args, vae, image_path, tensor_transform, device, weight_dtype):
    lr = Image.open(image_path).convert("RGB")
    resize_flag, new_width, new_height, ori_width, ori_height = preprocess_image(
        lr, args.upscale, args.process_size)
    pixel_values = tensor_transform(lr).unsqueeze(
        0).to(device, dtype=weight_dtype)
    pixel_values = torch.nn.functional.interpolate(
        pixel_values, size=(new_height, new_width), mode="bicubic", align_corners=False)
    pixel_values = pixel_values * 2 - 1
    model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
    return model_input


# -------------------------------
# 主流程
# -------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.mixed_precision)

    # 加载模型
    transformer, vae = load_models(args, device, weight_dtype)

    # 初始化收集器
    collector_val = ActivationCollector(
        max_samples_per_layer=200,
        max_qq_points=args.max_qq_points,
        seed=args.seed,
    )
    collector_ch = ChannelAnomalyCollector(topk_ratio=0.01)

    # 准备嵌入和时间步
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=device
    ).to(device, dtype=weight_dtype)
    timesteps = torch.tensor(
        [args.timestep], device=device, dtype=weight_dtype)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    # 处理两个数据集
    datasets = {
        "A": (args.dataset_A, "RealSR"),
        "B": (args.dataset_B, "DrealSR"),
    }

    for tag, (path, desc) in datasets.items():
        images = list_images(path, max_num=args.max_images_per_set)
        print(f"[{desc}] Found {len(images)} images")

        handles = register_hooks(transformer, collector_val, collector_ch, tag)

        for img_path in tqdm(images, desc=f"Processing {desc}"):
            try:
                model_input = image_to_latent(
                    args, vae, img_path, tensor_transform, device, weight_dtype)
                # 进行一次前向推理，激活值自动被 hook 收集
                _ = tile_sample(
                    model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                    latent_tiled_size=args.latent_tiled_size,
                    latent_tiled_overlap=args.latent_tiled_overlap,
                )
            except Exception as e:
                print(f"Error on {img_path}: {e}")
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # 移除当前数据集的 hooks（下一次循环重新注册）
        for h in handles:
            h.remove()

    # -------------- 计算指标 --------------
    final_report = {"args": vars(args), "layers": {}}

    for layer_name in sorted(collector_val.data.keys()):
        setA_vals = collector_val.data[layer_name].get("A", [])
        setB_vals = collector_val.data[layer_name].get("B", [])
        stats_A = compute_layer_stats(setA_vals) if setA_vals else None
        stats_B = compute_layer_stats(setB_vals) if setB_vals else None

        # KS 统计量：合并两个数据集的所有向量做 KS 检验
        ks_stat = None
        if setA_vals and setB_vals:
            all_A = np.concatenate(setA_vals)
            all_B = np.concatenate(setB_vals)
            try:
                from scipy.stats import ks_2samp
                ks_stat = float(ks_2samp(all_A, all_B).statistic)
            except Exception:
                pass

        # 通道异常频率
        ch_info = collector_ch.compute(layer_name)

        # 策略建议
        strategy = "uniform_quantization"  # default
        kurt_A = stats_A["kurtosis_excess"] if stats_A else 0
        kurt_B = stats_B["kurtosis_excess"] if stats_B else 0
        max_kurt = max(kurt_A, kurt_B)
        avg_var = ch_info["avg_var"] if ch_info else None
        ks_val = ks_stat if ks_stat is not None else 0.0

        if avg_var is not None and avg_var > 0.01:
            strategy = "SmoothQuant (static channel outliers)"
        else:
            if max_kurt > 10.0:
                strategy = "ECDF mapping"
        if ks_val > 0.1:
            strategy += " + online updating"

        final_report["layers"][layer_name] = {
            "setA": stats_A,
            "setB": stats_B,
            "ks_statistic": ks_stat,
            "channel_anomaly": ch_info,
            "strategy": strategy,
        }

    # 额外汇总 top 层
    top_kurt = sorted(
        final_report["layers"].items(),
        key=lambda kv: max(
            ((kv[1].get("setA") or {}).get("kurtosis_excess", -1e9)),
            ((kv[1].get("setB") or {}).get("kurtosis_excess", -1e9)),
        ),
        reverse=True,
    )[:10]
    top_ks = sorted(final_report["layers"].items(),
                    key=lambda kv: kv[1].get("ks_statistic") or 0.0,
                    reverse=True)[:10]
    final_report["top_heavy_tail_layers"] = [
        {
            "layer": k,
            "max_kurt": max(
                ((v.get("setA") or {}).get("kurtosis_excess", 0.0)),
                ((v.get("setB") or {}).get("kurtosis_excess", 0.0)),
            ),
        }
        for k, v in top_kurt
    ]
    final_report["top_input_dependent_layers"] = [
        {"layer": k, "ks_statistic": v["ks_statistic"]} for k, v in top_ks
    ]

    out_json = os.path.join(args.output_dir, "activation_preanalysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to {out_json}")
    print("Summary of strategies:")
    for name, info in final_report["layers"].items():
        if "ECDF" in info["strategy"] or "SmoothQuant" in info["strategy"]:
            print(f"  {name}: {info['strategy']}")


if __name__ == "__main__":
    main()
