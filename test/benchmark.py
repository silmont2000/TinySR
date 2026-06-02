# fmt:off
'''
# --------------------------------------------------------------------------------
#    script modified from  (https://github.com/Microtreei/TSD-SR)
# --------------------------------------------------------------------------------
'''
import time
import os
import sys
import json as _json
sys.path.append(".")
import argparse
import torch
import torch.nn.functional as tfF
import numpy as np
from peft import LoraConfig
from diffusers import (
    StableDiffusion3Pipeline,
)
from models.tinysr.stage1_defaults import (
    CKPT, DEFAULT_PYRAMID_CONFIG, LORA_R, SMOKE_RANK_PATTERN, make_lora_config,
    DEFAULT_TIMESTEP,
)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.tinysr.pyramid_config import PyramidArchConfig
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel
from models.vae.autoencoder_tiny  import  AutoencoderTiny
from models.quant.tiler import tile_sample

from utils.vaehook import _init_tiled_vae
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from utils.util import load_lora_state_dict_warn,load_lora_state_dict

from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['p','t'],default="p", help='p for pyramid or t for tinysr')

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="path/to/your/model", help='path to the pretrained sd3')
    parser.add_argument("--vae_path", type=str, default="path/to/your/vae", help='path to tsd-sr lora weights')
    parser.add_argument("--lora_dir", type=str, default="path/to/your/lora", help='path to tsd-sr lora weights')
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models", help='cache directory for downloading models')
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/", help='path to prompt embeddings')

    parser.add_argument("--batch_size", type=int, default=1, help='batch size for benchmark')
    parser.add_argument("--num_iterations", type=int, default=100, help='number of benchmark iterations')

    parser.add_argument("--rank", type=int, default=64, help='rank for transformer')
    parser.add_argument("--rank_vae", type=int, default=64, help='rank for vae')

    parser.add_argument("--is_use_tile", type=bool, default=False, help='whether to use tiled vae')
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224, help='tiled size for tiled vae decoder')
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024, help='tiled size for tiled vae encoder')
    parser.add_argument("--latent_tiled_size", type=int, default=64, help='tiled size for transformer latent')
    parser.add_argument("--latent_tiled_overlap", type=int, default=8, help='tiled overlap for transformer latent')

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    return parser.parse_args()


def main(args, pixel_values):
    with torch.no_grad():
        pixel_values = pixel_values.to(args.device, dtype=weight_dtype)
        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        model_input = model_input.to(args.device, dtype=weight_dtype)

        if args.model=='p':
            output, pre_last = transformer(
                hidden_states=model_input, timestep=timesteps,
                pooled_projections=pooled_prompt_embeds, return_dict=False,
            )
            last_grid_hw = transformer.pyramid_config.p_states[-1].grid_hw
            if pyramid_loss_type == "a":
                scale_factor = last_grid_hw // transformer.pyramid_config.p_states[0].grid_hw
                input_up = tfF.interpolate(model_input, scale_factor=scale_factor,
                                            mode='bilinear', align_corners=False)
                denoised = input_up - output
            else:
                latent_before = transformer._tokens_to_latent(pre_last, last_grid_hw)
                denoised = latent_before - output

        elif args.model=='t':
            model_pred = tile_sample(model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                                    latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap)
            denoised = model_input - model_pred

        image = vae.decode(denoised / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
        return image, denoised


if __name__ == "__main__":
    args = parse_args()

    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_HUB_CACHE'] = args.cache_dir

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    torch.manual_seed(args.seed)

    # Load the pretrained models
    if args.model=='p':
        pc_path = os.path.join(args.lora_dir, "pyramid_config.json")
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"pyramid_config.json not found in {args.lora_dir}")
        with open(pc_path) as f:
            pc = PyramidArchConfig.from_dict(_json.load(f))
        print(f"  pyramid: sample_size={pc.sample_size}, {[(s.num_blocks,s.dim,s.grid_hw) for s in pc.p_states]}")
        transformer = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
            args.pretrained_model_name_or_path, pyramid_config=pc,
            subfolder="transformer", torch_dtype=weight_dtype,
        )
    elif args.model=='t':
        transformer = TinySD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer", torch_dtype=weight_dtype, low_cpu_mem_usage=False, ignore_mismatched_sizes=True, cache_dir=args.cache_dir)

    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)

    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    if args.lora_dir and args.model=='p':
        rp_path = os.path.join(args.lora_dir, "rank_pattern.json")
        if os.path.exists(rp_path):
            with open(rp_path) as f:
                rank_pattern = _json.load(f)
        else:
            rank_pattern = None
        transformer_lora_config = make_lora_config(rank_pattern=rank_pattern)

        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()

        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="model.safetensors", cache_dir=args.cache_dir)
        transformer_lora_state_dict = dict(transformer_lora_state_dict)
        load_lora_state_dict_warn(transformer_lora_state_dict, transformer)
        if transformer_lora_state_dict:
            print(f"  Warning: {len(transformer_lora_state_dict)} LoRA keys not loaded:")
            sample_keys = list(transformer_lora_state_dict.keys())[:3]
            for k in sample_keys:
                print(f"    {k}")


        # ── Pyramid: load loss_type for denoising ──
        pyramid_loss_type = "a"
        tc_path = os.path.join(args.lora_dir, "train_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                pyramid_loss_type = _json.load(f).get("loss_type", "a")
        print(f"  pyramid loss_type={pyramid_loss_type}")

    elif args.lora_dir and args.model=='t':
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0","proj","linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()

        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="transformer.safetensors", cache_dir=args.cache_dir)
        load_lora_state_dict(transformer_lora_state_dict, transformer)



    vae = vae.to(args.device, dtype=weight_dtype)
    transformer = transformer.to(args.device, dtype=weight_dtype)
    if args.model=='p':
        param_cnt = sum(p.numel() for p in (transformer.p_states).parameters())
    elif args.model=='t':
        param_cnt = sum(p.numel() for p in transformer.transformer_blocks.parameters() )

    print("#Param.", param_cnt/1e6, "M")

    # Sample timestep for each image
    timesteps = torch.tensor([1000.], device=args.device, dtype=weight_dtype)

    # Load the prompt embeddings
    pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.expand(args.batch_size, -1)

    if args.model=='p':
        H, W = 256, 256
    elif args.model=='t':
        H, W = 512, 512
    total_time = 0.0
    mem_records = []

    # Warmup: 5 forward passes on dummy data for GPU warmup
    torch.cuda.empty_cache()
    transformer.eval()
    warmup_pixels = torch.randn(args.batch_size, 3, H, W, device=args.device, dtype=weight_dtype)
    for _ in range(20):
        main(args, warmup_pixels)
    print("Warmup done (20 iters).")

    for i in range(args.num_iterations):
        pixel_values = torch.randn(args.batch_size, 3, H, W, device=args.device, dtype=weight_dtype)

        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        image, denoised = main(args, pixel_values)
        torch.cuda.synchronize()
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        total_time += (end_time - start_time)
        mem_records.append(peak_mem)

        torch.cuda.empty_cache()

    if args.model=='p':
        param_cnt = sum(p.numel() for p in (transformer.p_states).parameters())
    elif args.model=='t':
        param_cnt = sum(p.numel() for p in transformer.transformer_blocks.parameters() )

    mem_arr = np.array(mem_records)
    print(f"#Param. {param_cnt/1e6:.2f} M")
    print(f"Resolution: {H}x{W}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Average time: {total_time / args.num_iterations:.4f} sec/batch")
    print(f"Average time per sample: {total_time / (args.num_iterations * args.batch_size):.6f} sec/sample")
    if len(mem_arr) > 0:
        print(f"Peak mem  avg: {np.mean(mem_arr):.0f} MB")
        print(f"Peak mem  max: {np.max(mem_arr):.0f} MB")
