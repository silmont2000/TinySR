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
from models.quant.calibrate import calibrate_all_layers
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
    parser.add_argument("--enable_hadamard_rotate", action="store_true")
    parser.add_argument("--hadamard_mode", type=str, default="random_orthogonal",
                        choices=["random_orthogonal", "fast_hadamard"],
                        help="Rotation mode for --enable_hadamard_rotate. "
                             "random_orthogonal: dense QR, O(n²), any size. "
                             "fast_hadamard: Walsh-Hadamard with power-of-2 pad, O(n log n).")
    parser.add_argument("--align_nunchaku_inference", action="store_true",
                        help="After calibration, switch QuantLinearW4A4 forward to nunchaku-aligned "
                             "path (dynamic per-group act quant + per-group residual + unsmoothed lora). "
                             "Makes train_quant validation output match nunchaku inference exactly.")
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--save_quant_meta", action="store_true")
    parser.add_argument("--save_nunchaku", type=str, default="__auto__",
                        help="After calibration+freeze, convert SVDQ low-rank+quantized layers "
                             "to nunchaku compatible safetensors and save to this path. "
                             "Default: <output_dir>/nunchaku.safetensors. Pass empty string to disable.")
    parser.add_argument("--save_merged_backbone", type=str, default="__auto__",
                        help="After merge_and_unload(LoRA), save the complete merged backbone "
                             "to a directory (creates config.json + model.safetensors). "
                             "Default: <output_dir>/merged_backbone. Pass empty string to disable.")
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


def save_nunchaku_safetensors(transformer, output_path: str):
    from safetensors.torch import save_file

    try:
        from nunchaku.lora.flux.packer import NunchakuWeightPacker
    except ImportError:
        raise ImportError(
            "nunchaku is required for --save_nunchaku. "
            "Install from the whl: nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl"
        )

    packer = NunchakuWeightPacker(bits=4)
    group_size = 64
    eps = 1e-8

    state_dict = {}
    layer_count = 0

    for name, m in transformer.named_modules():
        if not isinstance(m, QuantLinearW4A4):
            continue
        wq = m.weight_quantizer
        if not hasattr(wq, "branch") or wq.branch is None:
            continue

        out_features = m.weight.shape[0]
        in_features = m.weight.shape[1]
        rank = wq.rank
        if rank <= 0:
            continue

        smooth_scale = wq.smooth_scale
        if smooth_scale is None:
            smooth_scale = torch.ones(in_features, device=m.weight.device, dtype=m.weight.dtype)
        else:
            smooth_scale = smooth_scale.detach().to(device=m.weight.device, dtype=m.weight.dtype)

        branch_a = wq.branch.a.weight.detach()  # (rank, in_features)
        branch_b = wq.branch.b.weight.detach()  # (out_features, rank)

        low_rank = branch_b @ branch_a
        smoothed_weight = m.weight.detach() * smooth_scale.reshape(1, -1)
        residual_for_nunchaku = (smoothed_weight - low_rank) / smooth_scale.reshape(1, -1)
        # Per-group int4 quantization — use GPTQ if raw calibration inputs available
        assert in_features % group_size == 0, \
            f"[nunchaku] in_features ({in_features}) must be divisible by group_size ({group_size}) for layer {name}"

        raw_cache = getattr(wq, "raw_input_cache", None)
        raw_samples = sum(t.shape[0] for t in raw_cache) if raw_cache else 0
        gptq_used = (raw_samples > 0)
        if gptq_used:
            from models.quant.ops import gptq_per_group_int4
            raw_inputs = torch.cat(raw_cache, dim=0).to(
                device=residual_for_nunchaku.device, dtype=torch.float32)
            qweight_int, per_group_scale = gptq_per_group_int4(
                residual_for_nunchaku, raw_inputs, group_size=group_size, bits=4,
                symmetric=True, block_size=128, damp_percentage=0.01, eps=eps)
            per_group_scale = per_group_scale.to(dtype=m.weight.dtype)
            qweight_int = qweight_int.to(torch.int32)
        else:
            residual_fp = residual_for_nunchaku.float()
            residual_groups = residual_fp.view(out_features, in_features // group_size, group_size)
            per_group_scale = residual_groups.abs().amax(dim=-1).clamp_min(eps) / 7.0
            qweight_int = torch.round(residual_groups / per_group_scale.unsqueeze(-1)).clamp_(-8, 7).to(torch.int32)
            qweight_int = qweight_int.view(out_features, in_features).contiguous()

        # Pack qweight: int32 → int8 packed layout
        packed_qweight = packer.pack_weight(qweight_int)

        # Pack wscales: input (out, in//group_size) → output (in//group_size, out) in nunchaku layout
        per_group_scale_fp = per_group_scale.contiguous().to(dtype=m.weight.dtype)  # (out, in//group_size)
        packed_wscales = packer.pack_scale(per_group_scale_fp, group_size=group_size)

        # Pack proj_down / proj_up
        # pack_lowrank_weight(down=True) handles the (rank,in) → (in,rank) transpose internally.
        # The quantize kernel computes lora_act = RAW_x @ proj_down (no smooth division),
        # so we must "unsmooth" proj_down by dividing by smooth_scale.
        # This matches deepcompressor's converter: lora_down.div_(smooth.unsqueeze(0))
        proj_down_raw = branch_a.contiguous().to(dtype=torch.float32)  # (rank, in_features)
        proj_down_raw = proj_down_raw / smooth_scale.float().reshape(1, -1).clamp_min(eps)
        proj_down_raw = proj_down_raw.to(dtype=m.weight.dtype)
        proj_up_raw = branch_b.contiguous().to(dtype=m.weight.dtype)     # (out_features, rank)
        packed_proj_down = packer.pack_lowrank_weight(proj_down_raw, down=True)
        packed_proj_up = packer.pack_lowrank_weight(proj_up_raw, down=False)

        state_dict[f"{name}.qweight"] = packed_qweight.cpu()
        state_dict[f"{name}.wscales"] = packed_wscales.cpu()
        state_dict[f"{name}.proj_down"] = packed_proj_down.cpu()
        state_dict[f"{name}.proj_up"] = packed_proj_up.cpu()
        smooth_factor_cpu = torch.ones(in_features, dtype=torch.float16)
        state_dict[f"{name}.smooth_factor"] = smooth_factor_cpu
        state_dict[f"{name}.smooth_factor_orig"] = smooth_factor_cpu.clone()
        if hasattr(m, "hadamard_signs") and m.hadamard_signs is not None:
            state_dict[f"{name}.hadamard_rotated"] = torch.tensor([1], dtype=torch.int32)
        if m.bias is not None:
            state_dict[f"{name}.bias"] = m.bias.detach().cpu().to(torch.float16)

        print(
            f"  [nunchaku] {name}: "
            f"qweight={tuple(packed_qweight.shape)} "
            f"wscales={tuple(packed_wscales.shape)} "
            f"pdown={tuple(packed_proj_down.shape)} "
            f"pup={tuple(packed_proj_up.shape)} "
            f"rank={rank} bias={'Y' if m.bias is not None else 'N'} "
            f"{'GPTQ('+str(raw_samples)+')' if gptq_used else 'MINMAX'}"
        )
        layer_count += 1

    if layer_count == 0:
        print("[nunchaku] WARNING: no SVDQ layers found — empty safetensors saved")

    # Save rotation matrices (global, shared by all rotated layers)
    # Save rotation info (global for random_orthogonal, per-layer for fast_hadamard)
    rot_info = getattr(transformer, "_hadamard_rotation_info", None)
    if rot_info:
        if rot_info["mode"] == "random_orthogonal":
            for size, Q in rot_info["matrices"].items():
                state_dict[f"_rotation.{size}"] = Q.cpu().contiguous()
        elif rot_info["mode"] == "fast_hadamard":
            for layer_name, signs in rot_info["signs_map"].items():
                state_dict[f"{layer_name}.hadamard_signs"] = signs.cpu()
                state_dict[f"{layer_name}.hadamard_padded"] = torch.tensor(
                    [rot_info["padded_map"][layer_name]], dtype=torch.int32)

    save_file(state_dict, output_path)
    print(f"[nunchaku] saved {layer_count} layers ({len(state_dict)} tensors) -> {output_path}")


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
    if args.save_merged_backbone == "__auto__":
        args.save_merged_backbone = os.path.join(args.output_dir, "merged_backbone")
    if args.save_nunchaku == "__auto__":
        args.save_nunchaku = os.path.join(args.output_dir, "nunchaku.safetensors")

    if args.align_nunchaku_inference:
        if args.weight_group_size < 0:
            args.weight_group_size = 64
        if args.act_group_size < 0:
            args.act_group_size = 64

    # Shared runtime data
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location="cpu",
    ).to(device=device, dtype=weight_dtype)
    timesteps = torch.tensor(
        [args.timestep], device=device, dtype=weight_dtype)

    # Model preparation
    transformer, vae = load_models(
        args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
        args.rank, args.cache_dir, device, weight_dtype, skip_lora=False)
    transformer = transformer.merge_and_unload()
    if args.save_merged_backbone:
        import json
        from safetensors.torch import save_file
        out_dir = args.save_merged_backbone
        os.makedirs(out_dir, exist_ok=True)
        weight_path = os.path.join(out_dir, "diffusion_pytorch_model.safetensors")
        save_file(transformer.state_dict(), weight_path)
        config_path = os.path.join(out_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(dict(transformer.config), f, indent=2)
        print(f"[MERGED] saved merged backbone ({len(transformer.state_dict())} keys) -> {out_dir}/")
    replaced_layers = replace_quant_layers(
        transformer, args.quant_scope, args.quant_config,
        args.w_bits, args.a_bits, args.svdq_rank, args.svdq_smooth_alpha,
        svdq_iterations=args.svdq_iterations, act_group_size=args.act_group_size,
        weight_group_size=args.weight_group_size)

    if args.enable_hadamard_rotate:
        from models.quant.hadamard import enable_rotation
        rotation_info = enable_rotation(transformer, mode=args.hadamard_mode)
        transformer._hadamard_rotation_info = rotation_info

    if args.quant_config:
        print("[INFO] freezing with config-provided per-layer params (no calibration)")
        calibrate_all_layers(
            transformer, do_search=False, compute_error=False)
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

    if args.align_nunchaku_inference and args.quant_scope != "none":
        from models.quant.layers import QuantLinearW4A4
        for name, m in transformer.named_modules():
            if isinstance(m, QuantLinearW4A4):
                m.enable_nunchaku_aligned()
        print(f"[ALIGN] nunchaku-aligned inference enabled on QuantLinearW4A4 layers")

    if args.save_nunchaku and args.quant_scope != "none":
        save_nunchaku_safetensors(transformer, args.save_nunchaku)

    quant_meta = collect_quant_meta(
        transformer) if args.quant_scope != "none" else []

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
                quant_meta, timing)


if __name__ == "__main__":
    main()
