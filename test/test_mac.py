import glob
import time
import os
import sys
sys.path.append(".")
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig
from diffusers import (
    StableDiffusion3Pipeline,
)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny  import  AutoencoderTiny
from thop import profile, clever_format

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="your model", help='path to the pretrained sd3')
    parser.add_argument("--lora_dir", type=str, default="your lora", help='path to tsd-sr lora weights')
    parser.add_argument("--vae_path", type=str, default="your vae", help='path to tsd-sr lora weights')

    parser.add_argument("--embedding_dir", type=str, default="dataset/default/", help='path to prompt embeddings')

    parser.add_argument("--rank", type=int, default=64, help='rank for transformer')
    parser.add_argument("--rank_vae", type=int, default=64, help='rank for vae')

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4, help='upscale factor')
    parser.add_argument("--process_size", type=int, default=512, help='process size for images')
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain', help='color alignment method')

    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--quant", action="store_true", help="enable W4A8 quantization (requires --calib_cache)")
    parser.add_argument("--w_bits", type=int, default=4, help="weight quantization bits")
    parser.add_argument("--a_bits", type=int, default=8, help="activation quantization bits")
    parser.add_argument("--svdq_rank", type=int, default=32, help="SVD low-rank branch rank")
    parser.add_argument("--smooth_alpha", type=float, default=0.99, help="smooth quant migration strength")
    parser.add_argument("--quant_scope", type=str, choices=["ffn_only", "attn_only", "dit_full"], default="attn_only")
    parser.add_argument("--joint_qkv", action="store_true", help="enable joint QKV low-rank branch")
    parser.add_argument("--calib_cache", type=str, default=None, help="path to calibration cache .pt file")

    return parser.parse_args()

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])


class TinySR(nn.Module):
    def __init__(self):
        super().__init__()
        transformer = TinySD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer",
                                            torch_dtype=weight_dtype, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
                                            cache_dir=args.cache_dir)
        vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)
        self.timesteps = torch.tensor([1000.]).to(device=args.device, dtype=weight_dtype)
        self.vae = vae.to(device=args.device, dtype=weight_dtype)
        self.transformer = transformer.to(args.device, dtype=weight_dtype)

        self.prompt_embeds = torch.load(os.path.join(args.embedding_dir, "prompt_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
        self.pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)

        if args.lora_dir and args.lora_dir != "your lora":
            from utils.util import load_lora_state_dict
            transformer_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
            )
            transformer.add_adapter(transformer_lora_config)
            transformer.enable_adapters()
            transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
                args.lora_dir, weight_name="transformer.safetensors", cache_dir=args.cache_dir)
            load_lora_state_dict(transformer_lora_state_dict, transformer)

    def forward(self, pixel_values, size):
        with torch.no_grad():
            pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode='bicubic', align_corners=False)
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to(args.device, dtype=weight_dtype).clamp(-1,1)

            model_input = self.vae.encode(pixel_values).latents * self.vae.config.scaling_factor
            model_input = model_input.to(args.device, dtype=weight_dtype)

            model_pred =  self.transformer(
                        hidden_states=model_input,
                        timestep=self.timesteps,
                        pooled_projections=self.pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
            latent_stu = model_input - model_pred

            image = self.vae.decode(latent_stu / self.vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)
        return image


if __name__ == "__main__":
    args = parse_args()
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    model = TinySR()

    if args.quant:
        from train.quant_w4a4 import (
            QuantLinearW4A4,
            LowRankBranch,
            LowRankAffineQuantComponent,
            replace_linear_with_w4a4,
            replace_qkv_with_joint_w4a4,
            set_quant_enabled,
            set_observer_enabled,
            load_calib_cache,
            freeze_quant_params,
        )

        FFN_SUFFIXES = ["ff.net.0.proj.base_layer", "ff.net.2.base_layer",
                        "ff.net.0.proj", "ff.net.2"]
        ATTN_SUFFIXES = ["attn.to_q.base_layer", "attn.to_k.base_layer",
                         "attn.to_v.base_layer", "attn.to_out.0.base_layer",
                         "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
        EXTRA_SUFFIXES = ["proj_out.base_layer", "proj_out"]

        scope_map = {
            "ffn_only": FFN_SUFFIXES,
            "attn_only": ATTN_SUFFIXES,
            "dit_full": FFN_SUFFIXES + ATTN_SUFFIXES + EXTRA_SUFFIXES,
        }
        target_suffixes = scope_map[args.quant_scope]

        weight_quant_kwargs = {
            "bits": args.w_bits,
            "symmetric": True,
            "per_channel": True,
            "ch_axis": 0,
            "rank": args.svdq_rank,
            "smooth_alpha": args.smooth_alpha,
        }
        act_quant_kwargs = {
            "bits": args.a_bits,
            "symmetric": True,
            "per_channel": False,
        }

        replaced = replace_linear_with_w4a4(
            model.transformer,
            target_suffixes=target_suffixes,
            skip_keywords=("lora_",),
            weight_quant_kind="svdq",
            weight_quant_kwargs=weight_quant_kwargs,
            act_quant_kwargs=act_quant_kwargs,
        )
        print(f"[QUANT] replaced {len(replaced)} linear layers")

        if args.joint_qkv:
            joint_replaced = replace_qkv_with_joint_w4a4(
                model.transformer,
                weight_quant_kind="svdq",
                act_quant_kind="affine",
                weight_quant_kwargs=weight_quant_kwargs,
                act_quant_kwargs=act_quant_kwargs,
            )
            if joint_replaced:
                print(f"[QUANT] joint QKV: {len(joint_replaced)//3} blocks")

        if args.calib_cache and os.path.exists(args.calib_cache):
            for m in model.transformer.modules():
                if isinstance(m, QuantLinearW4A4):
                    wq = m.weight_quantizer
                    if isinstance(wq, LowRankAffineQuantComponent) and wq.rank > 0:
                        wq.branch = LowRankBranch(
                            m.in_features, m.out_features,
                            rank=wq.rank, alpha=wq.alpha
                        ).to(device=m.weight.device, dtype=m.weight.dtype)

            load_calib_cache(model.transformer, args.calib_cache)
            print(f"[QUANT] loaded calib cache: {args.calib_cache}")

            if args.joint_qkv:
                import re
                block_mods = {}
                for name, m in model.transformer.named_modules():
                    if isinstance(m, QuantLinearW4A4):
                        match = re.match(r'(transformer_blocks\.\d+)\.attn\.to_([qkv])\.base_layer', name)
                        if match:
                            blk, proj = match.group(1), match.group(2)
                            block_mods.setdefault(blk, {})[proj] = m

                for blk, pmap in block_mods.items():
                    if {'q', 'k', 'v'} <= pmap.keys():
                        qb = pmap['q'].weight_quantizer.branch
                        kb = pmap['k'].weight_quantizer.branch
                        vb = pmap['v'].weight_quantizer.branch
                        if qb and kb and vb and qb.rank > 0:
                            shared_a = qb.a
                            del kb.a
                            del vb.a
                            kb.a = shared_a
                            vb.a = shared_a
        else:
            print("[QUANT] no calib cache — running simplified freeze (MACs-only)")
            freeze_quant_params(model.transformer)
            set_quant_enabled(model.transformer, True)
            print("[QUANT] freeze done")

    input_shape_multi1 = (1, 3, 128, 128)
    num_iterations = 100

    for _ in range(num_iterations):
        dummy_input_multi1 = torch.randn(input_shape_multi1).to(device=args.device, dtype=weight_dtype)
        dummy_input_multi2 = (512, 512)

        model(dummy_input_multi1, dummy_input_multi2)


    param_cnt = sum(p.numel() for p in model.parameters() )
    print("#Param.", param_cnt/1e6, "M")

    total_macs = 0
    total_params = 0
    num_iterations = 1
    print(f"Calculating MACs and Parameters for {num_iterations} iterations...")

    for _ in range(num_iterations):
        dummy_input_multi1 = torch.randn(input_shape_multi1).to(device=args.device, dtype=weight_dtype)
        dummy_input_multi2 = (512, 512)

        macs, params = profile(model, inputs=(dummy_input_multi1, dummy_input_multi2), verbose=False)

        total_macs += macs
        total_params += params

    average_macs = total_macs / num_iterations
    average_params = total_params / num_iterations

    average_macs_formatted, average_params_formatted = clever_format([average_macs, average_params], "%.6f")

    print(f"\n--- Average Results over {num_iterations} Iterations ---")
    print(f"Multi-input Model Average MACs: {average_macs_formatted}")
    print(f"Multi-input Model Average Parameters: {average_params_formatted}")
