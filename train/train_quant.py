# fmt:off
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.quant.layers import QuantLinearW4A4, collect_quant_meta, set_quant_enabled, set_observer_enabled
from models.quant.calibration import load_calib_cache, save_calib_cache
from models.quant.freeze import freeze_quant_params
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
    parser.add_argument("--act_group_size", type=int, default=64,
                        help="Per-group size for activation quantization. -1 = per-tensor. "
                             "Default 64 matches nunchaku SVDQ-W4A4.")
    parser.add_argument("--weight_group_size", type=int, default=-1,
                        help="Per-group size for weight residual quantization. -1 = per-channel GPTQ. "
                             "Set to 64 to match nunchaku.")
    parser.add_argument("--quant_config", type=str, default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--svdq_rank", type=int, default=32)

    # --- smooth_alpha 三级优先 ---
    # 最高优：从历史报告加载逐层alpha
    parser.add_argument("--load_smooth_alpha_report", type=str, default=None,
                        help="Path to w4a4_report.json from a prior run. "
                             "Loads per-layer smooth_alpha overrides. "
                             "Overrides --svdq_smooth_alpha and --search_smooth_alpha. "
                             "No error computation / no cascade.")
    # 次高优：所有层统一alpha
    parser.add_argument("--svdq_smooth_alpha", type=float, default=0.5,
                        help="Fixed smooth_alpha for ALL layers. "
                             "Overridden by --load_smooth_alpha_report. "
                             "When set (without --search_smooth_alpha), no error computation.")
    # 最低优：搜索最优alpha
    parser.add_argument("--search_smooth_alpha", nargs="?", const="grid",
                        choices=["grid", "cascade"], default=None,
                        help="Search optimal smooth_alpha per layer. "
                             "'grid' (default): single-pass grid search. "
                             "'cascade': layer-by-layer cascade freeze. "
                             "Overridden by --load_smooth_alpha_report and --svdq_smooth_alpha.")
    # ---------------------------------------------------
    parser.add_argument("--cascade_calib_images", type=int, default=4,
                        help="Number of calibration images used in cascade forward passes "
                             "(only relevant with --search_smooth_alpha cascade).")
    parser.add_argument("--svdq_alpha_grid", type=int, default=11,
                        help="Number of grid points for alpha search.")
    parser.add_argument("--svdq_iterations", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="adain")
    parser.add_argument("--quant_scope", type=str, choices=["none", "ffn_only", "attn_only", "dit_full"],
                        default="ffn_only")
    parser.add_argument("--calib_images", type=int, default=8)
    parser.add_argument("--calib_cache", type=str, default=None)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--save_quant_meta", action="store_true")
    parser.add_argument("--save_model_path", nargs="?", const="__auto__", default=None)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--save_quant_state", type=str, default=None,
                        help="Save full quantized model state_dict after calibration+freeze. "
                             "Includes all weights, quant params, smooth_scale, branch weights. "
                             "Load back with --load_quant_state to skip calibration entirely.")
    parser.add_argument("--load_quant_state", type=str, default=None,
                        help="Load a previously saved quantized state_dict. "
                             "Replaces Linear layers, loads all weights/quant params, "
                             "disables observers, enables quantizers. Skips calibration.")
    parser.add_argument("--analyze_activation", action="store_true")
    parser.add_argument("--analyze_max_samples", type=int, default=200)
    parser.add_argument("--analyze_max_points", type=int, default=20000)

    return parser.parse_args()

# fmt:on


def _derive_output_dir(args):
    scope_tag = args.quant_scope
    if args.search_smooth_alpha == "cascade":
        alpha_tag = "cascade"
    elif args.search_smooth_alpha:
        alpha_tag = "sa"
    else:
        alpha_tag = str(int(args.svdq_smooth_alpha * 100))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/w{args.w_bits}a{args.a_bits}_svdq_r{args.svdq_rank}_{scope_tag}_a{alpha_tag}_{ts}"


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

    if args.quant_config:
        print(
            f"[INFO] quant_config mode: {args.quant_config} — skipping calibration, using config values")
        calib_image_names = []
    else:
        calib_image_names = get_image_names(args.calib_input_dir)
        if len(calib_image_names) == 0:
            raise RuntimeError(
                f"No calibration images found in {args.calib_input_dir}")
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
    timesteps = torch.tensor(
        [args.timestep], device=device, dtype=weight_dtype)

    # Model preparation
    #   Branch A: load saved state_dict → skip all calibration
    #   Branch B: load torchao export → TorchAOQuantLinear modules（invalid）
    #   Branch C: calibration + freeze
    if args.load_quant_state:
        transformer, vae = load_models(
            args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
            args.rank, args.cache_dir, device, weight_dtype, skip_lora=False)
        transformer = transformer.merge_and_unload()
        replaced_layers = replace_quant_layers(
            transformer, args.quant_scope, args.quant_config,
            args.w_bits, args.a_bits, args.svdq_rank, args.svdq_smooth_alpha,
            svdq_iterations=args.svdq_iterations, act_group_size=args.act_group_size,
            weight_group_size=args.weight_group_size)
        missing, unexpected = transformer.load_state_dict(
            torch.load(args.load_quant_state, map_location="cpu"), strict=False)
        print(f"[QUANT] quant state loaded <- {args.load_quant_state}")
        if missing:
            print(
                f"[QUANT]   missing keys (from backbone, expected): {len(missing)}")
        if unexpected:
            print(f"[QUANT]   unexpected keys: {len(unexpected)}")
        quant_meta = collect_quant_meta(
            transformer) if args.quant_scope != "none" else []

    else:
        transformer, vae = load_models(
            args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
            args.rank, args.cache_dir, device, weight_dtype, skip_lora=False)
        transformer = transformer.merge_and_unload()
        replaced_layers = replace_quant_layers(
            transformer, args.quant_scope, args.quant_config,
            args.w_bits, args.a_bits, args.svdq_rank, args.svdq_smooth_alpha,
            svdq_iterations=args.svdq_iterations, act_group_size=args.act_group_size,
            weight_group_size=args.weight_group_size)

        if args.quant_config:
            print(
                "[INFO] freezing with config-provided per-layer params (no calibration)")
            freeze_quant_params(
                transformer, search_smooth_alpha=False, compute_error=False)
        elif args.quant_scope != "none" and args.calib_cache and os.path.exists(args.calib_cache):
            load_calib_cache(transformer, args.calib_cache)
        else:
            calibrate_w4a4(
                transformer, vae, calib_image_names,
                pooled_prompt_embeds, timesteps, weight_dtype,
                quant_scope=args.quant_scope,
                calib_images=args.calib_images,
                search_mode=args.search_smooth_alpha,
                cascade_calib_images=args.cascade_calib_images,
                load_smooth_alpha_report=args.load_smooth_alpha_report,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap,
                device=args.device, upscale=args.upscale, process_size=args.process_size,
                alpha_grid_size=args.svdq_alpha_grid,
            )
            if args.calib_cache:
                save_calib_cache(transformer, args.calib_cache)

        if args.save_quant_state and args.quant_scope != "none":
            quant_state = {}
            for name, m in transformer.named_modules():
                if isinstance(m, QuantLinearW4A4):
                    for k, v in m.state_dict().items():
                        quant_state[f"{name}.{k}"] = v
            torch.save(quant_state, args.save_quant_state)
            print(
                f"[QUANT] quant state saved ({len(quant_state)} keys) -> {args.save_quant_state}")

        quant_meta = collect_quant_meta(
            transformer) if args.quant_scope != "none" else []
        save_path = args.save_model_path

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
