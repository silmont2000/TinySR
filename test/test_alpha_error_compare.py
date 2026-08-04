"""
Compare output-space reconstruction error: sweep_tier-2-3 (alpha=1.0) vs
custom alpha values on the SAME calibration images.

The w4a4_report.json stores WEIGHT-space errors (svd_error, quant_error).
This script computes OUTPUT-space MSE so you can directly compare with
the alpha search results from train_quant.py.

Usage:
    conda activate tinysr_nunchaku
    python test/test_alpha_error_compare.py \
        --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr \
        --vae_path checkpoint/vae/separable \
        --lora_dir checkpoint/tinysr \
        --calib_input_dir ../testset/RealSR/LR \
        --calib_images 100 \
        --quant_scope attn_only --quant_ffn_blocks "1,6,9,10,11" \
        --rank 64
"""
import argparse
import gc
import os
import sys

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.layers import (
    parse_ffn_blocks,
    set_observer_enabled,
    set_quant_enabled,
)
from models.quant.inference import replace_quant_layers
from models.quant.calibrate import _eval_quant_error
from models.quant.tiler import tile_sample
from models.pipeline import image_to_latent, get_image_names
from utils.util import load_lora_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--calib_input_dir", type=str, required=True)
    parser.add_argument("--calib_images", type=int, default=100)
    parser.add_argument("--quant_scope", type=str, choices=["ffn_only", "attn_only", "dit_full"], default="attn_only")
    parser.add_argument("--quant_ffn_blocks", type=str, default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_layer", type=str, default=None,
                        help="Only evaluate a specific layer (e.g. 'transformer_blocks.0.attn.to_q')")
    return parser.parse_args()


def collect_calib_data(transformer, vae, calib_names, pooled_prompt_embeds, timesteps,
                       weight_dtype, device, upscale, process_size):
    """Phase 1: collect act_absmax for each quant layer from calibration images."""
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    set_quant_enabled(transformer, False)
    set_observer_enabled(transformer, True)

    for image_path in tqdm(calib_names, desc="Collecting calib stats"):
        model_input, _ = image_to_latent(
            upscale, process_size, vae, image_path, tensor_transform, device, weight_dtype)
        tile_sample(model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                    latent_tiled_size=64, latent_tiled_overlap=8)

    set_observer_enabled(transformer, False)


def main():
    args = parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("[1/5] Loading backbone...")
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)

    print("[2/5] Loading LoRA and merging...")
    if args.lora_dir:
        transformer_lora_config = LoraConfig(
            r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            args.lora_dir, weight_name="transformer.safetensors", cache_dir=args.cache_dir)
        load_lora_state_dict(transformer_lora_state_dict, transformer)
        transformer = transformer.merge_and_unload()

    print("[3/5] Replacing Linear -> QuantLinearW4A4...")
    replaced = replace_quant_layers(
        transformer,
        quant_scope=args.quant_scope,
        w_bits=4, a_bits=4,
        svdq_rank=args.rank,
        svdq_smooth_alpha=0.5,
        svdq_iterations=0,
        ffn_blocks=parse_ffn_blocks(args.quant_ffn_blocks),
    )
    print(f"  Replaced {len(replaced)} layers")

    transformer = transformer.to(device, dtype=weight_dtype).eval()
    vae = vae.to(device, dtype=weight_dtype).eval()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"),
        map_location=device,
    ).to(dtype=weight_dtype)

    print("[4/5] Collecting calibration statistics...")
    calib_names = get_image_names(args.calib_input_dir)[:args.calib_images]
    print(f"  Images: {len(calib_names)}")

    collect_calib_data(transformer, vae, calib_names, pooled_prompt_embeds, timesteps,
                       weight_dtype, device, args.upscale, args.process_size)

    print("[5/5] Evaluating output-space MSE...")
    print()

    alphas_to_test = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]

    header = (f"{'Layer':<50} | {'a=1.0(it0)':>14} | {'best_a it0':>10} | {'best_err it0':>14} | "
              f"{'a=1.0(it10)':>14} | {'best_a it10':>10} | {'best_err it10':>14} | "
              f"{'a=1.0(it100)':>14} | {'best_a it100':>10} | {'best_err it100':>14} | {'inp_max':>10} | {'inp_std':>10}")
    print(header)
    print("-" * len(header))

    results = []

    for record in replaced:
        layer_name = record["name"] if isinstance(record, dict) else record

        if args.target_layer and args.target_layer not in layer_name:
            continue

        try:
            m = transformer.get_submodule(layer_name)
        except AttributeError:
            continue

        if not hasattr(m, "weight_quantizer"):
            continue

        wq = m.weight_quantizer

        if wq.act_absmax is None or len(wq.input_cache) == 0:
            continue

        weight = m.weight
        act_absmax = wq.act_absmax.to(device=weight.device, dtype=weight.dtype).clamp_min(1e-8)
        weight_absmax = weight.detach().abs().amax(dim=0).clamp_min(1e-8)

        # Match cascade's exact parameters
        aq = m.act_quantizer
        act_bits = aq.quantizer.bits
        act_sym = aq.quantizer.symmetric
        act_group_size = getattr(aq.quantizer, "group_size", -1)

        inputs_cat = torch.cat([t.cuda() for t in wq.input_cache], dim=0)
        inp_max = inputs_cat.abs().max().item()
        inp_std = inputs_cat.std().item()

        def eval_err(alpha, num_iters):
            return _eval_quant_error(
                weight, act_absmax, weight_absmax,
                wq, inputs_cat, alpha,
                act_bits=act_bits, act_symmetric=act_sym,
                act_group_size=act_group_size,
                num_iterations=num_iters,
            )

        err_1_0_it0 = eval_err(1.0, 0)
        err_1_0_it10 = eval_err(1.0, 10)
        err_1_0_it100 = eval_err(1.0, 100)

        best_alpha_it0, best_err_it0 = 1.0, err_1_0_it0.item()
        best_alpha_it10, best_err_it10 = 1.0, err_1_0_it10.item()
        best_alpha_it100, best_err_it100 = 1.0, err_1_0_it100.item()

        for alpha in alphas_to_test:
            e0 = eval_err(alpha, 0).item()
            e10 = eval_err(alpha, 10).item()
            e100 = eval_err(alpha, 100).item()
            if e0 < best_err_it0:
                best_err_it0, best_alpha_it0 = e0, alpha
            if e10 < best_err_it10:
                best_err_it10, best_alpha_it10 = e10, alpha
            if e100 < best_err_it100:
                best_err_it100, best_alpha_it100 = e100, alpha

        short_name = layer_name if len(layer_name) < 50 else "..." + layer_name[-47:]
        print(f"{short_name:<50} | {err_1_0_it0.item():14.6e} | {best_alpha_it0:10.2f} | {best_err_it0:14.6e} | {err_1_0_it10.item():14.6e} | {best_alpha_it10:10.2f} | {best_err_it10:14.6e} | {err_1_0_it100.item():14.6e} | {best_alpha_it100:10.2f} | {best_err_it100:14.6e} | {inp_max:10.4f} | {inp_std:10.4f}")

        results.append({
            "layer": layer_name,
            "err_alpha_1.0_it0": err_1_0_it0.item(),
            "best_alpha_it0": best_alpha_it0,
            "best_err_it0": best_err_it0,
            "err_alpha_1.0_it10": err_1_0_it10.item(),
            "best_alpha_it10": best_alpha_it10,
            "best_err_it10": best_err_it10,
            "err_alpha_1.0_it100": err_1_0_it100.item(),
            "best_alpha_it100": best_alpha_it100,
            "best_err_it100": best_err_it100,
            "inp_max": inp_max,
            "inp_std": inp_std,
        })

    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("  it0   = num_iterations=0   (matches sweep_tier-2-3 minmax)")
    print("  it10  = num_iterations=10  (matches cascade search_iters cap)")
    print("  it100 = num_iterations=100 (matches current train_quant full GPTQ)")
    print()
    print("  If best_a at it10 != 1.0: the search_iters=10 cap causes alpha shift")
    print()
    print("  cascade calib images=100 | standalone calib images=100")
    print("  Same RealSR LR images in both cases")
    print("=" * 80)


if __name__ == "__main__":
    main()
