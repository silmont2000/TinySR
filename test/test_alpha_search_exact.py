"""
Exact reproduction of cascade alpha search for a specific layer.
Calls search_alpha with the same parameters as _resolve_alpha in calibrate.py.

Usage:
    python test/test_alpha_search_exact.py \
        --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr \
        --vae_path checkpoint/vae/separable \
        --lora_dir checkpoint/tinysr \
        --calib_input_dir ../testset/RealSR/LR \
        --calib_images 100 \
        --quant_scope attn_only --quant_ffn_blocks "1,6,9,10,11" \
        --rank 64 \
        --target_layer "transformer_blocks.0.attn.to_q"
"""
import argparse
import gc
import os
import sys

import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.layers import parse_ffn_blocks, set_observer_enabled, set_quant_enabled
from models.quant.inference import replace_quant_layers
from models.quant.calibrate import search_alpha, compute_smooth_scale, _eval_quant_error
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
    parser.add_argument("--target_layer", type=str, required=True)
    parser.add_argument("--alpha_grid", type=int, default=21)
    parser.add_argument("--svdq_iterations", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading backbone...")
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        torch_dtype=weight_dtype, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)

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

    print("Replacing layers...")
    replaced = replace_quant_layers(
        transformer, quant_scope=args.quant_scope, w_bits=4, a_bits=4,
        svdq_rank=args.rank, svdq_smooth_alpha=0.5,
        svdq_iterations=args.svdq_iterations,
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
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=device,
    ).to(dtype=weight_dtype)

    print("Collecting calibration statistics (Phase 1)...")
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    calib_names = get_image_names(args.calib_input_dir)[:args.calib_images]
    print(f"  Images: {len(calib_names)}")

    set_quant_enabled(transformer, False)
    set_observer_enabled(transformer, True)
    for image_path in tqdm(calib_names, desc="Phase 1"):
        model_input, _ = image_to_latent(
            args.upscale, args.process_size, vae, image_path, tensor_transform, device, weight_dtype)
        tile_sample(model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                    latent_tiled_size=64, latent_tiled_overlap=8)
    set_observer_enabled(transformer, False)

    # --- Exact reproduction of _resolve_alpha for the target layer ---
    m = transformer.get_submodule(args.target_layer)
    wq = m.weight_quantizer
    aq = m.act_quantizer

    # Line 130-131 of calibrate.py
    act_absmax = wq.act_absmax.to(device=m.weight.device, dtype=m.weight.dtype).clamp_min(1e-8)
    w_absmax = m.weight.detach().abs().amax(dim=0).clamp_min(1e-8)

    # Line 134-141
    act_bits = aq.quantizer.bits
    act_sym = aq.quantizer.symmetric
    act_scale_val = aq.quantizer.scale.detach().clone()
    act_group_size = getattr(aq.quantizer, "group_size", -1)

    input_cache = wq.input_cache
    search_iters = max(0, min(args.svdq_iterations, 10))

    print(f"\n{'='*80}")
    print(f"Target layer: {args.target_layer}")
    print(f"  act_bits={act_bits}, act_sym={act_sym}, act_scale={act_scale_val.item()}, act_group_size={act_group_size}")
    print(f"  svdq_iterations={args.svdq_iterations}, search_iters={search_iters}")
    print(f"  alpha_grid={args.alpha_grid}")
    print(f"  input_cache tokens: {sum(t.shape[0] for t in input_cache)}")
    print(f"{'='*80}")

    # ---- Search 1: Cascade-style (fixed act_scale) ----
    print(f"\n[CASCADE-STYLE] search_alpha with act_scale={act_scale_val.item():.4f}")
    print(f"{'Alpha':>8} | {'Error':>14}")
    print("-" * 30)

    alpha_grid = [i / (args.alpha_grid - 1) for i in range(args.alpha_grid)]

    best_alpha_cascade = None
    best_err_cascade = float("inf")
    errors_cascade = {}

    for alpha in alpha_grid:
        err = _eval_quant_error(
            weight=m.weight, act_absmax=act_absmax, weight_absmax=w_absmax,
            weight_quantizer=wq, inputs=torch.cat(input_cache, dim=0).to(device=m.weight.device, dtype=m.weight.dtype),
            alpha=alpha, act_bits=act_bits, act_symmetric=act_sym,
            act_scale=act_scale_val, act_group_size=act_group_size,
            num_iterations=search_iters,
        )
        err_val = err.item()
        errors_cascade[alpha] = err_val
        marker = "<<< BEST" if err_val < best_err_cascade else ""
        if err_val < best_err_cascade:
            best_err_cascade = err_val
            best_alpha_cascade = alpha
        print(f"{alpha:8.4f} | {err_val:14.6e}{marker}")

    print(f"\n  CASCADE-STYLE best: alpha={best_alpha_cascade:.4f}, err={best_err_cascade:.6e}")

    # ---- Search 2: Dynamic scale ----
    print(f"\n[DYNAMIC-SCALE] search_alpha with act_scale=None")
    print(f"{'Alpha':>8} | {'Error':>14}")
    print("-" * 30)

    best_alpha_dyn = None
    best_err_dyn = float("inf")
    for alpha in alpha_grid:
        err = _eval_quant_error(
            weight=m.weight, act_absmax=act_absmax, weight_absmax=w_absmax,
            weight_quantizer=wq, inputs=torch.cat(input_cache, dim=0).to(device=m.weight.device, dtype=m.weight.dtype),
            alpha=alpha, act_bits=act_bits, act_symmetric=act_sym,
            act_scale=None, act_group_size=act_group_size,
            num_iterations=search_iters,
        )
        err_val = err.item()
        marker = "<<< BEST" if err_val < best_err_dyn else ""
        if err_val < best_err_dyn:
            best_err_dyn = err_val
            best_alpha_dyn = alpha
        print(f"{alpha:8.4f} | {err_val:14.6e}{marker}")

    print(f"\n  DYNAMIC-SCALE best: alpha={best_alpha_dyn:.4f}, err={best_err_dyn:.6e}")

    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Cascade-style (act_scale={act_scale_val.item():.4f}): best alpha={best_alpha_cascade:.4f}, err={best_err_cascade:.6e}")
    print(f"  Dynamic-scale  (act_scale=None):             best alpha={best_alpha_dyn:.4f}, err={best_err_dyn:.6e}")
    if abs(best_alpha_cascade - best_alpha_dyn) > 0.01:
        print(f"  >>> MISMATCH: act_scale causes alpha shift of {abs(best_alpha_cascade - best_alpha_dyn):.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
