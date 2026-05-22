# fmt:off
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.quant.inference import (
    get_image_names,
    get_weight_dtype,
    load_models,
    replace_quant_layers,
    preprocess_one_image,
    tile_sample,
)
from models.quant.layers import set_quant_enabled, set_observer_enabled
from models.quant.components import LowRankBranch
from models.vae.autoencoder_tiny import AutoencoderTiny


def parse_args():
    parser = argparse.ArgumentParser(description="W4A4 quantized inference timing benchmark.")

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--input_dir", type=str, default="dataset/StableSR_testsets/DIV2K_V2_val/test_LR")

    parser.add_argument("--load_quant_state", type=str, required=True,
                        help="Path to a previously saved quantized state_dict (.pt). "
                             "Produced by --save_quant_state during calibration.")
    parser.add_argument("--quant_scope", type=str, choices=["ffn_only", "attn_only", "dit_full"],
                        default="dit_full")
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--svdq_rank", type=int, default=32)
    parser.add_argument("--svdq_smooth_alpha", type=float, default=0.5)

    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations before timing.")
    parser.add_argument("--num_images", type=int, default=0,
                        help="Number of images to time (0 = all).")
    parser.add_argument("--skip_decode", action="store_true",
                        help="Skip VAE decode in timing (measure forward-only).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save decoded images to this directory (I/O excluded from timing).")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank (must match calibration).")

    parser.add_argument("--int4_cuda", action="store_true",
                        help="Use WMMA int8 Tensor Core kernel for true int4 matmul.")

    return parser.parse_args()


def main(args, pixel_values, new_height, new_width, transformer, vae, timesteps, pooled_prompt_embeds, weight_dtype):
    with torch.no_grad():
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=(new_height, new_width), mode='bicubic', align_corners=False)
        pixel_values = pixel_values * 2 - 1
        pixel_values = pixel_values.to(args.device, dtype=weight_dtype)
        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        model_input = model_input.to(args.device, dtype=weight_dtype)
        model_pred = tile_sample(model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                                 latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap)
        latent_stu = model_input - model_pred
        image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1, 1)
        return image


if __name__ == "__main__":
    args = parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.mixed_precision)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    print(f"[bench] device={args.device}  dtype={args.mixed_precision}  scope={args.quant_scope}")
    print(f"[bench] loading model ...")
    transformer, vae = load_models(
        args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
        args.rank, args.cache_dir, device, weight_dtype, skip_lora=False)

    print(f"[bench] replacing Linear → QuantLinearW4A4 ...")
    replaced = replace_quant_layers(
        transformer, args.quant_scope, None,
        args.w_bits, args.a_bits, args.svdq_rank, args.svdq_smooth_alpha)
    print(f"[bench]   {len(replaced)} layers replaced")

    print(f"[bench] loading state_dict <- {args.load_quant_state}")
    state = torch.load(args.load_quant_state, map_location="cpu")

    # Pass 1: load matching keys via standard state_dict
    model_sd = transformer.state_dict()
    matched = {k: v for k, v in state.items() if k in model_sd and v.shape == model_sd[k].shape}
    missing, unexpected = transformer.load_state_dict(matched, strict=False)

    # Pass 2: handle shape-mismatched keys by walking module tree
    # (quantizer scales go from scalar→per-channel after calibration)
    reshaped = 0
    not_found = 0
    for k, v in state.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            continue
        if k not in model_sd:
            not_found += 1
            continue
        target = transformer
        *module_path, attr_name = k.rsplit('.', 1) if '.' in k else (k,)
        if module_path:
            try:
                for part in module_path[0].split('.'):
                    target = getattr(target, part)
            except AttributeError:
                not_found += 1
                continue
        v = v.to(device=model_sd[k].device, dtype=model_sd[k].dtype)
        for pname, param in target.named_parameters(recurse=False):
            if pname == attr_name:
                param.data = v
                reshaped += 1
                break
        else:
            for bname, buf in target.named_buffers(recurse=False):
                if bname == attr_name:
                    target._buffers[attr_name] = v
                    reshaped += 1
                    break

    # Pass 3: direct injection into QuantLinearW4A4 layers
    # residual / smooth_scale / act_absmax are registered as buffers with
    # value=None at construction → they are excluded from state_dict().
    # branch is a plain attribute = None → also absent from state_dict().
    # These keys exist in the saved checkpoint but NOT in the fresh model's
    # state_dict, so passes 1&2 silently skip them. This pass walks
    # the replaced list to inject them directly into each QuantLinearW4A4.
    quant_injected = 0
    for rec in replaced:
        name = rec["name"]
        m = transformer.get_submodule(name)
        prefix = name + ".weight_quantizer."
        device_v = m.weight.device
        dtype_v = m.weight.dtype
        wq = m.weight_quantizer

        r_key = prefix + "residual"
        if r_key in state:
            wq.residual = state[r_key].to(device=device_v, dtype=dtype_v)
            quant_injected += 1

        s_key = prefix + "smooth_scale"
        if s_key in state and state[s_key] is not None:
            wq.smooth_scale = state[s_key].to(device=device_v, dtype=dtype_v)

        a_key = prefix + "act_absmax"
        if a_key in state and state[a_key] is not None:
            wq.act_absmax = state[a_key].to(device=device_v, dtype=dtype_v)

        b_a_key = prefix + "branch.a.weight"
        b_b_key = prefix + "branch.b.weight"
        if b_a_key in state and b_b_key in state:
            in_f = m.in_features
            out_f = m.out_features
            rank = wq.rank
            if rank > 0:
                branch = LowRankBranch(in_f, out_f, rank=rank, alpha=wq.alpha, weight=None)
                branch.to(device=device_v, dtype=dtype_v)
                branch.a.weight.data.copy_(state[b_a_key].to(device=device_v, dtype=dtype_v))
                branch.b.weight.data.copy_(state[b_b_key].to(device=device_v, dtype=dtype_v))
                wq.branch = branch

    # Remove quantizer keys from not_found count — they were handled by pass 3
    not_found_skipped = 0
    for k in list(state.keys()):
        if k not in model_sd and ".weight_quantizer." in k:
            not_found_skipped += 1

    if reshaped:
        print(f"[bench]   shape-changed (direct assign): {reshaped}")
    if not_found - not_found_skipped:
        print(f"[bench]   not in model (skipped): {not_found - not_found_skipped}")

    # Free state dict references — they hold parameter tensors alive
    del model_sd

    if missing:
        print(f"[bench]   missing keys (backbone, expected): {len(missing)}")
    if unexpected:
        print(f"[bench]   unexpected keys (saved but not in model): {len(unexpected)}")
    if quant_injected:
        print(f"[bench]   quantizer state injected (residual/branch): {quant_injected} layers")

    # Pass 4: inject act_quantizer scales from state_dict (required by int4 CUDA)
    act_loaded = 0
    for rec in replaced:
        name = rec["name"]
        m = transformer.get_submodule(name)
        aq_key = name + ".act_quantizer.quantizer.scale"
        if aq_key in state:
            m.act_quantizer.quantizer.scale = state[aq_key].to(device=m.weight.device)
            m.act_quantizer.quantizer.calibrated = True
            act_loaded += 1
    if act_loaded:
        print(f"[bench]   act scales injected: {act_loaded} layers")

    # Optionally switch to int4 dequant + cuBLAS fp16 path
    if args.int4_cuda:
        from models.quant.int4_pack import pack_all_quant_layers
        n_packed = pack_all_quant_layers(transformer)
        if n_packed:
            print(f"[bench]   int4 dequant + cuBLAS fp16 enabled: {n_packed} layers")
        else:
            print("[bench]   int4 CUDA skipped (no layers packed)")


    # Release FP16 residuals from CUDA cache so they don't inflate peak mem
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"[bench]   model memory (weights+buffers): {model_mem:.0f} MB")

    set_observer_enabled(transformer, False)
    set_quant_enabled(transformer, True)

    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location="cpu",
    ).to(device=device, dtype=weight_dtype)
    timesteps = torch.tensor([args.timestep], device=device, dtype=weight_dtype)

    image_names = get_image_names(args.input_dir)
    if len(image_names) == 0:
        raise RuntimeError(f"No images found in {args.input_dir}")
    print(f"[bench] found {len(image_names)} images")

    # Select images for timing
    n_warmup = max(0, args.warmup)
    n_timed = args.num_images if args.num_images > 0 else max(len(image_names) - n_warmup, 0)
    total_needed = max(n_warmup , n_timed)
    if total_needed > len(image_names):
        n_timed = min(n_timed,len(image_names))
        n_warmup = min(n_warmup,len(image_names))
        total_needed = len(image_names)

    used = image_names[:total_needed]
    print(f"[bench] warmup={n_warmup}  timed={n_timed}  images={len(used)}/{len(image_names)}")

    param_cnt = sum(p.numel() for p in transformer.parameters())
    print(f"[bench] #Param. {param_cnt/1e6:.1f}M")
    # ---- Output dir ----
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/timing_w{args.w_bits}a{args.a_bits}_r{args.svdq_rank}_{args.quant_scope}_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[bench] output -> {args.output_dir}")
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    # config = Int4WeightOnlyConfig(
    #     group_size=32,
    #     # int4_packing_format="tile_packed_to_4d",
    #     # int4_choose_qparams_algorithm="hqq",
    # )
    # transformer = transformer.eval().to(torch.bfloat16).to("cuda")
    # transformer = torch.compile(transformer, mode="max-autotune")
    # quantize_(transformer, config)

    # ---- Warmup ----
    if n_warmup > 0:
        print("[bench] warming up ...")
        for i, image_path in enumerate(tqdm(used[:n_warmup], desc="Warmup"), 1):
            lr = Image.open(image_path).convert('RGB')
            resize_flag, new_width, new_height, ori_width, ori_height = preprocess_one_image(lr, args.upscale, args.process_size)
            pixel_values = tensor_transform(lr).unsqueeze(0).to(device, dtype=weight_dtype)
            main(args, pixel_values, new_height, new_width, transformer, vae, timesteps, pooled_prompt_embeds, weight_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- Timing ----
    print(f"[bench] timing {n_timed} images ...")
    times = []
    mem_records = []

    for image_path in tqdm(used[:n_timed], desc="Timing"):
        # I/O + resize logic — outside timing (same as tinysr L250-270)
        lr = Image.open(image_path).convert('RGB')
        resize_flag, new_width, new_height, ori_width, ori_height = preprocess_one_image(lr, args.upscale, args.process_size)
        pixel_values = tensor_transform(lr).unsqueeze(0).to(device, dtype=weight_dtype)

        # torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        # --- TIMING START ---
        decoded = main(args, pixel_values, new_height, new_width, transformer, vae, timesteps, pooled_prompt_embeds, weight_dtype)
        # --- TIMING END ---
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        times.append(elapsed)
        mem_records.append(peak_mem)
        # Save decoded image AFTER timing (I/O excluded from measurement)
        if decoded is not None and args.output_dir:
            image = decoded
            image_pil = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
            if resize_flag:
                image_pil = image_pil.resize((
                    int(ori_width * args.upscale),
                    int(ori_height * args.upscale)))
            save_path = os.path.join(args.output_dir, os.path.basename(image_path))
            image_pil.save(save_path)


    # ---- Report ----
    if not times:
        print("[bench] no timed images — nothing to report")
    else:
        arr = np.array(times)
        mem = np.array(mem_records)
        print()
        print("=" * 55)
        print(f"  images timed      : {len(times)}")
        print(f"  warmup            : {n_warmup}")
        print(f"  quant_scope       : {args.quant_scope}")
        print(f"  w{args.w_bits}a{args.a_bits} r={args.svdq_rank}")
        print(f"  tiled={args.latent_tiled_size},{args.latent_tiled_overlap}")
        print(f"  decode            : {'skipped' if args.skip_decode else 'included'}")
        print("-" * 55)
        print(f"  avg  (ms)         : {np.mean(arr) * 1000:.1f}")
        print(f"  p50  (ms)         : {np.percentile(arr, 50) * 1000:.1f}")
        print(f"  p90  (ms)         : {np.percentile(arr, 90) * 1000:.1f}")
        print(f"  p95  (ms)         : {np.percentile(arr, 95) * 1000:.1f}")
        print(f"  min  (ms)         : {np.min(arr) * 1000:.1f}")
        print(f"  max  (ms)         : {np.max(arr) * 1000:.1f}")
        print(f"  std  (ms)         : {np.std(arr) * 1000:.1f}")
        print("-" * 55)
        print(f"  peak mem avg (MB) : {np.mean(mem):.0f}")
        print(f"  peak mem p50 (MB) : {np.percentile(mem, 50):.0f}")
        print(f"  peak mem max (MB) : {np.max(mem):.0f}")
        print("=" * 55)
