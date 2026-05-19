import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.quant.layers import collect_quant_meta
from models.quant.calibration import load_calib_cache, save_calib_cache
from models.quant.inference import (
    calibrate_w4a4,
    get_image_names,
    get_weight_dtype,
    load_models,
    replace_quant_layers,
    run_inference,
    save_report,
)
from models.quant.analysis import ActivationErrorAnalyzer
from models.quant.serialization import export_torchao_model, load_torchao_model


def parse_args():
    parser = argparse.ArgumentParser(description="W4A4 quantized inference test for TinySR.")

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--input_dir", type=str, default="dataset/test_image/")
    parser.add_argument("--calib_input_dir", type=str,
                        default="dataset/StableSR_testsets/DrealSRVal_crop128/test_LR")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--quant_config", type=str, default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--svdq_rank", type=int, default=32)
    parser.add_argument("--svdq_smooth_alpha", type=float, default=0.5)
    parser.add_argument("--svdq_no_error", action="store_true", dest="svdq_no_error")
    parser.add_argument("--search_smooth_alpha", action="store_true")
    parser.add_argument("--layer_cascade_smooth_alpha", action="store_true")
    parser.add_argument("--svdq_alpha_grid", type=int, default=7)
    parser.add_argument("--cascade_calib_images", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="adain")
    parser.add_argument("--quant_scope", type=str, choices=["none", "ffn_only", "attn_only", "dit_full"],
                        default="ffn_only")
    parser.add_argument("--calib_images", type=int, default=8)
    parser.add_argument("--calib_cache", type=str, default=None)
    parser.add_argument("--load_smooth_alpha_report", type=str, default=None)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--save_quant_meta", action="store_true")
    parser.add_argument("--save_model_path", nargs="?", const="__auto__", default=None)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--analyze_activation", action="store_true")
    parser.add_argument("--analyze_max_samples", type=int, default=200)
    parser.add_argument("--analyze_max_points", type=int, default=20000)

    return parser.parse_args()


def _derive_output_dir(args):
    scope_tag = args.quant_scope
    if args.layer_cascade_smooth_alpha:
        alpha_tag = "layer_cascade"
    elif args.search_smooth_alpha:
        alpha_tag = "sa"
    else:
        alpha_tag = str(int(args.svdq_smooth_alpha * 100))
    return f"outputs/w{args.w_bits}a{args.a_bits}_svdq_r{args.svdq_rank}_{scope_tag}_a{alpha_tag}"


def main():
    args = parse_args()

    # Environment
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.mixed_precision)

    # Image discovery
    image_names = get_image_names(args.input_dir)
    if len(image_names) == 0:
        raise RuntimeError(f"No input images found in {args.input_dir}")
    calib_image_names = get_image_names(args.calib_input_dir)
    if len(calib_image_names) == 0:
        raise RuntimeError(f"No calibration images found in {args.calib_input_dir}")
    print(f"[INFO] images: {len(image_names)}")
    print(f"[INFO] calib_images: {len(calib_image_names)}")
    print(f"[INFO] quant_scope: {args.quant_scope}")

    if args.output_dir is None:
        args.output_dir = _derive_output_dir(args)
    print(f"[INFO] output_dir: {args.output_dir}")

    # Shared runtime data
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location="cpu",
    ).to(device=device, dtype=weight_dtype)
    timesteps = torch.tensor([args.timestep], device=device, dtype=weight_dtype)

    # Model preparation (branch A: torchao; branch B: calibration)
    if args.load_model_path:
        transformer, vae = load_models(
            args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
            args.rank, args.cache_dir, device, weight_dtype, skip_lora=True)
        if args.lora_dir:
            print("[torchao] --lora_dir ignored: LoRA is baked into exported quantized weights")
        replaced_layers, quant_meta, _ = load_torchao_model(
            transformer, args.load_model_path, device=device)
    else:
        transformer, vae = load_models(
            args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
            args.rank, args.cache_dir, device, weight_dtype, skip_lora=False)
        replaced_layers = replace_quant_layers(
            transformer, args.quant_scope, args.quant_config,
            args.w_bits, args.a_bits, args.svdq_rank, args.svdq_smooth_alpha)

        if args.quant_scope != "none" and args.calib_cache and os.path.exists(args.calib_cache):
            load_calib_cache(transformer, args.calib_cache)
        else:
            calibrate_w4a4(
                transformer, vae, calib_image_names,
                pooled_prompt_embeds, timesteps, weight_dtype,
                quant_scope=args.quant_scope,
                calib_images=args.calib_images,
                search_smooth_alpha=args.search_smooth_alpha,
                layer_cascade_smooth_alpha=args.layer_cascade_smooth_alpha,
                cascade_calib_images=args.cascade_calib_images,
                svdq_no_error=args.svdq_no_error,
                load_smooth_alpha_report=args.load_smooth_alpha_report,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap,
                device=args.device, upscale=args.upscale, process_size=args.process_size,
                alpha_grid_size=args.svdq_alpha_grid,
            )
            if args.calib_cache:
                save_calib_cache(transformer, args.calib_cache)

        quant_meta = collect_quant_meta(transformer) if args.quant_scope != "none" else []
        save_path = args.save_model_path
        if save_path == "__auto__":
            save_path = os.path.join(args.output_dir, "torchao_model.pt")
        if save_path:
            export_torchao_model(transformer, save_path, replaced_layers, quant_meta,
                                 model_args=vars(args))

    # Activation analysis
    analyzer = None
    if args.analyze_activation and args.quant_scope != "none":
        analyzer = ActivationErrorAnalyzer(
            max_samples_per_layer=args.analyze_max_samples,
            max_points_per_sample=args.analyze_max_points)
        analyzer.register_hooks(transformer)

    # Inference
    timing = run_inference(
        transformer, vae, image_names,
        pooled_prompt_embeds, timesteps, weight_dtype,
        output_dir=args.output_dir,
        upscale=args.upscale, process_size=args.process_size,
        align_method=args.align_method, warmup_images=args.warmup_images,
        latent_tiled_size=args.latent_tiled_size,
        latent_tiled_overlap=args.latent_tiled_overlap,
        device=args.device,
    )

    # Teardown
    if analyzer is not None:
        analyzer.remove_hooks()
        analyzer.save_report(
            os.path.join(args.output_dir, "activation_error_report.json"),
            args_dict=vars(args))

    print(f"[TIME] avg={timing['avg_time_sec']} "
          f"p50={timing['p50_time_sec']} p90={timing['p90_time_sec']} "
          f"count={timing['timed_image_count']}")

    save_report(args.output_dir, vars(args), replaced_layers,
                quant_meta if args.save_quant_meta else [], timing)


if __name__ == "__main__":
    main()
