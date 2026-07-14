"""
Nunchaku SVDQ-W4A4 inference for TinySR.

Usage:
    python test/test_nunchaku_inference.py \\
        --pretrained_model_name_or_path outputs/merged_backbone \\
        --nunchaku_state outputs/nunchaku_fixed.safetensors \\
        --quant_scope dit_full --rank 32
"""
import argparse
import gc
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.layers import get_target_suffixes, parse_ffn_blocks
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


def parse_args():
    parser = argparse.ArgumentParser(description="Nunchaku W4A4 inference for TinySR.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--input_dir", type=str, default="dataset/test_image/")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--nunchaku_state", type=str, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--quant_scope", type=str, choices=["ffn_only", "attn_only", "dit_full"],
                        default="dit_full")
    parser.add_argument("--quant_ffn_blocks", type=str, default=None,
                        help="Comma-separated block indices to include FFN layers when quant_scope=attn_only.")
    parser.add_argument("--quant_exclude_keywords", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="adain")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--profile_nunchaku", action="store_true",
                        help="Enable per-layer nunchaku call counting and timing.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark mode: random pixel tensors, full pipeline (VAE encode + transformer + VAE decode).")
    parser.add_argument("--bench_iterations", type=int, default=100,
                        help="Number of benchmark iterations.")
    parser.add_argument("--bench_image_h", type=int, default=512,
                        help="Height of random pixel tensor for benchmark (VAE will produce derived latent size).")
    parser.add_argument("--bench_image_w", type=int, default=512,
                        help="Width of random pixel tensor for benchmark (VAE will produce derived latent size).")
    return parser.parse_args()


def _gaussian_weights(tile_width, tile_height, nbatches, device, in_channels, dtype):
    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    midpoint_y = tile_height / 2
    x_probs = [
        np.exp(-(x - midpoint_x) ** 2 / (tile_width * tile_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(tile_width)
    ]
    y_probs = [
        np.exp(-(y - midpoint_y) ** 2 / (tile_height * tile_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(tile_height)
    ]
    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device, dtype=dtype), (nbatches, in_channels, 1, 1))


@torch.no_grad()
def tile_sample(lq_latent, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                tile_size=64, tile_overlap=8):
    _, _, h, w = lq_latent.size()
    if h * w <= tile_size * tile_size:
        return transformer(
            hidden_states=lq_latent.to(dtype=weight_dtype),
            timestep=timesteps,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

    tile_size = min(tile_size, min(h, w))
    tile_weights = _gaussian_weights(
        tile_size, tile_size, 1, lq_latent.device,
        transformer.config.in_channels, weight_dtype,
    )

    grid_rows = 0
    cur_x = 0
    while cur_x < w:
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1
    grid_cols = 0
    cur_y = 0
    while cur_y < h:
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1

    input_list = []
    noise_preds = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_tile = lq_latent[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size]
            input_list.append(input_tile)

            if len(input_list) == 1 or col == grid_cols - 1:
                input_list_t = torch.cat(input_list, dim=0)
                model_out = transformer(
                    hidden_states=input_list_t.to(dtype=weight_dtype),
                    timestep=timesteps,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                input_list = []
            noise_preds.append(model_out)

    noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=weight_dtype)
    contributors = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=weight_dtype)
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            tile_idx = row * grid_cols + col
            noise_pred[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] += (
                noise_preds[tile_idx] * tile_weights
            )
            contributors[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] += tile_weights
    noise_pred /= contributors.clamp_min(1e-8)
    return noise_pred


def replace_linear_with_nunchaku(module, target_suffixes, exclude_keywords, rank, profile=False, padded_dims=None):
    from tinysd3_nunchaku_w4a4 import NunchakuSVDQLinear
    padded_dims = padded_dims or {}

    replaced = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, torch.nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        if exclude_keywords and any(kw in name for kw in exclude_keywords):
            continue

        in_features = padded_dims.get(name, child.in_features)
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = module.get_submodule(parent_name) if parent_name else module
        quant_linear = NunchakuSVDQLinear(
            in_features=in_features,
            out_features=child.out_features,
            rank=rank,
            bias=child.bias is not None,
            torch_dtype=child.weight.dtype,
            device=child.weight.device,
        )
        if profile:
            quant_linear.enable_profile = True
            quant_linear.debug_name = name

        setattr(parent, child_name, quant_linear)
        replaced.append(name)

    if profile:
        print(f"[NUNCHAKU] profiling enabled on {len(replaced)} layers")
    else:
        print(f"[NUNCHAKU] replaced {len(replaced)} nn.Linear -> NunchakuSVDQLinear")
    return replaced


def load_nunchaku_state(transformer, state_path):
    from safetensors.torch import load_file

    state = load_file(state_path)
    missing, unexpected = transformer.load_state_dict(state, strict=False)

    nunchaku_keys = [k for k in state if any(
        k.endswith(suffix) for suffix in
        (".qweight", ".wscales", ".proj_down", ".proj_up", ".smooth_factor", ".bias")
    )]
    print(f"[NUNCHAKU] loaded {len(nunchaku_keys)} nunchaku tensors from {state_path}")

    if missing:
        nunchaku_missing = [k for k in missing if any(
            k.endswith(s) for s in (".qweight", ".wscales", ".proj_down",
                                    ".proj_up", ".smooth_factor", ".bias")
        )]
        non_nunchaku_missing = len(missing) - len(nunchaku_missing)
        if nunchaku_missing:
            print(f"[NUNCHAKU]   WARNING: missing nunchaku keys ({len(nunchaku_missing)}): "
                  f"{nunchaku_missing[:5]}{'...' if len(nunchaku_missing) > 5 else ''}")
        if non_nunchaku_missing:
            print(f"[NUNCHAKU]   non-nunchaku missing (backbone, expected): {non_nunchaku_missing}")
    if unexpected:
        for k in unexpected:
            print(f"[NUNCHAKU]   unexpected key: {k}")
        print(f"[NUNCHAKU]   total unexpected: {len(unexpected)}")


def run_nunchaku_benchmark(args, transformer, vae, timesteps, pooled_prompt_embeds, weight_dtype, device):
    """Benchmark full pipeline: VAE encode → transformer → latent subtract → VAE decode."""
    image_h, image_w = args.bench_image_h, args.bench_image_w
    print(f"[BENCH] pixel: {args.batch_size} × 3 × {image_h//4}→{image_h} × {image_w//4}→{image_w}, "
          f"{args.bench_iterations} iterations")

    # Warmup
    print(f"[BENCH] warming up ({args.warmup} iters)...")
    for i in range(args.warmup):
        with torch.no_grad():
            pv = torch.randn(args.batch_size, 3, image_h // 4, image_w // 4,
                             device=device, dtype=weight_dtype)
            pv = torch.nn.functional.interpolate(pv, size=(image_h, image_w),
                                                  mode="bicubic", align_corners=False)
            pv = pv * 2 - 1
            mi = vae.encode(pv).latents * vae.config.scaling_factor
            mp = transformer(
                hidden_states=mi,
                timestep=timesteps,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            ls = mi - mp
            _ = vae.decode(ls / vae.config.scaling_factor, return_dict=False)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run
    print(f"[BENCH] running {args.bench_iterations} iterations...")
    times = []
    mem_records = []
    for i in tqdm(range(args.bench_iterations), desc="bench"):
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            pv = torch.randn(args.batch_size, 3, image_h // 4, image_w // 4,
                             device=device, dtype=weight_dtype)
            pv = torch.nn.functional.interpolate(pv, size=(image_h, image_w),
                                                  mode="bicubic", align_corners=False)
            pv = pv * 2 - 1
            start = time.time()
            mi = vae.encode(pv).latents * vae.config.scaling_factor
            mp = transformer(
                hidden_states=mi,
                timestep=timesteps,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            ls = mi - mp
            _ = vae.decode(ls / vae.config.scaling_factor, return_dict=False)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
            mem_records.append(torch.cuda.max_memory_allocated() / 1024**2)
        times.append(time.time() - start)

    avg_ms = sum(times) / len(times) * 1000
    avg_per_sample_ms = avg_ms / args.batch_size
    mem_arr = np.array(mem_records)
    print(f"[BENCH] avg: {avg_ms:.2f}ms/iter  ({avg_per_sample_ms:.2f}ms/sample)  "
          f"batch={args.batch_size}  image={image_h}×{image_w}")
    print(f"[BENCH] Peak mem  avg: {np.mean(mem_arr):.0f} MB")
    print(f"[BENCH] Peak mem  max: {np.max(mem_arr):.0f} MB")


if __name__ == "__main__":
    args = parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Output dir: use user-provided value, or auto-derive from pretrained path
    if args.output_dir is None:
        calib_dir = os.path.basename(os.path.dirname(args.pretrained_model_name_or_path))
        args.output_dir = f"outputs/tinysr_nunchaku_{calib_dir}"
    os.makedirs(args.output_dir, exist_ok=True)

    print("[NUNCHAKU] loading merged backbone ...")
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )
    vae = AutoencoderTiny.from_pretrained(
        args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)

    target_suffixes = get_target_suffixes(args.quant_scope,
                                          ffn_blocks=parse_ffn_blocks(args.quant_ffn_blocks))
    exclude_keywords = tuple(
        s.strip() for s in args.quant_exclude_keywords.split(",") if s.strip()
    )

    replaced = replace_linear_with_nunchaku(
        transformer, target_suffixes, exclude_keywords, args.rank,
        profile=args.profile_nunchaku)

    load_nunchaku_state(transformer, args.nunchaku_state)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformer = transformer.to(device, dtype=weight_dtype).eval()
    vae = vae.to(device, dtype=weight_dtype).eval()

    param_cnt = sum(p.numel() for p in transformer.transformer_blocks.parameters())
    print(f"#Param. {param_cnt / 1e6:.1f}M")
    if torch.cuda.is_available():
        print(f"[NUNCHAKU] after to(device): {torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated")

    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"),
        map_location=device,
    ).to(dtype=weight_dtype)

    if args.benchmark:
        run_nunchaku_benchmark(args, transformer, vae, timesteps, pooled_prompt_embeds, weight_dtype, device)
        exit(0)

    if os.path.isdir(args.input_dir):
        image_names = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            image_names.extend(glob.glob(os.path.join(args.input_dir, pattern)))
        image_names = sorted(image_names)
    else:
        image_names = [args.input_dir]

    datalen = len(image_names)
    print(f"image_num {datalen}")
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.output_dir+ts, exist_ok=True)

    tensor_transforms = transforms.Compose([transforms.ToTensor()])

    print(f"Warming up ({args.warmup} iters)...")
    w_img = Image.open(image_names[0]).convert("RGB")
    w_w, w_h = w_img.size
    w_nh = args.upscale * w_h - (args.upscale * w_h) % 8
    w_nw = args.upscale * w_w - (args.upscale * w_w) % 8
    w_lr = w_img.resize((int(w_w * args.upscale), int(w_h * args.upscale)))
    w_pv = tensor_transforms(w_lr).unsqueeze(0).to(device, dtype=weight_dtype)

    for _ in range(args.warmup):
        w_pv_interp = torch.nn.functional.interpolate(
            w_pv, size=(w_nh, w_nw), mode="bicubic", align_corners=False)
        w_pv_norm = w_pv_interp * 2 - 1
        w_mi = vae.encode(w_pv_norm).latents * vae.config.scaling_factor
        w_mp = tile_sample(
            w_mi, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
            tile_size=args.latent_tiled_size, tile_overlap=args.latent_tiled_overlap,
        )
        _ = w_mi - w_mp
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup done.")

    pbar = tqdm(total=datalen)
    total_time = 0.0
    img_idx = 0

    while img_idx < datalen:
        image_name = image_names[img_idx]
        lr = Image.open(image_name).convert("RGB")
        ori_width, ori_height = lr.size
        upscale = args.upscale
        process_size = args.process_size

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

        lr_scale = lr.resize((int(ori_width * upscale), int(ori_height * upscale)))
        pixel_values = tensor_transforms(lr_scale).unsqueeze(0).to(device, dtype=weight_dtype)

        pv_list = [pixel_values]
        batch_names = [image_name]
        batch_lrs = [lr_scale]
        batch_origs = [lr]
        for j in range(1, args.batch_size):
            ni = img_idx + j
            if ni >= datalen:
                break
            nxt_name = image_names[ni]
            nxt = Image.open(nxt_name).convert("RGB")
            nlr = nxt.resize((int(nxt.size[0] * upscale), int(nxt.size[1] * upscale)))
            npv = tensor_transforms(nlr).unsqueeze(0).to(device, dtype=weight_dtype)
            if npv.shape != pixel_values.shape:
                break
            pv_list.append(npv)
            batch_names.append(nxt_name)
            batch_lrs.append(nlr)
            batch_origs.append(nxt)

        B = len(pv_list)
        pv_batch = torch.cat(pv_list, dim=0)
        start_time = time.time()
        with torch.no_grad():
            pb = torch.nn.functional.interpolate(
                pv_batch, size=(new_height, new_width), mode="bicubic", align_corners=False)
            pb = pb * 2 - 1
            mi = vae.encode(pb).latents * vae.config.scaling_factor
            if mi.shape[-1] * mi.shape[-2] <= args.latent_tiled_size ** 2:
                mp = transformer(
                    hidden_states=mi, timestep=timesteps,
                    pooled_projections=pooled_prompt_embeds, return_dict=False)[0]
            else:
                mp = tile_sample(
                    mi, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                    tile_size=args.latent_tiled_size, tile_overlap=args.latent_tiled_overlap,
                )
            ls = mi - mp
            images = vae.decode(ls / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - start_time

        for k in range(B):
            out_img = transforms.ToPILImage()(images[k].cpu() / 2 + 0.5)
            if resize_flag:
                out_img = out_img.resize((int(ori_width * upscale), int(ori_height * upscale)))
            if args.align_method == "adain":
                out_img = adain_color_fix(target=out_img, source=batch_origs[k])
            elif args.align_method == "wavelet":
                out_img = wavelet_color_fix(target=out_img, source=batch_lrs[k])
            out_img.save(os.path.join(args.output_dir, os.path.basename(batch_names[k])))

        img_idx += B
        pbar.update(B)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()
    print(f"Average time: {total_time / max(datalen, 1):.4f}s")

    if args.profile_nunchaku:
        from tinysd3_nunchaku_w4a4 import summarize_nunchaku_profile
        stats = summarize_nunchaku_profile(transformer)
        print(f"[NUNCHAKU][PROFILE] layers: {stats['layers']}")
        print(f"[NUNCHAKU][PROFILE] called_layers: {stats['called_layers']}")
        print(f"[NUNCHAKU][PROFILE] total_calls: {stats['calls']}")
        print(f"[NUNCHAKU][PROFILE] backend_calls: {stats['backend_calls']}")
        print(f"[NUNCHAKU][PROFILE] fallback_calls: {stats['fallback_calls']}")
        print(f"[NUNCHAKU][PROFILE] backend_ms: {stats['backend_ms']:.1f}")
        if stats['calls']:
            print(f"[NUNCHAKU][PROFILE] backend_ratio: {stats['backend_calls']/stats['calls']:.4f}")
