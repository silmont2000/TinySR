# fmt:off
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import torch
import torch.nn as nn
from thop import clever_format
from safetensors.torch import load_file

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.layers import get_target_suffixes, parse_ffn_blocks

from tinysd3_nunchaku_w4a4 import (
    NunchakuSVDQLinear,
    replace_linear_with_nunchaku_w4a4,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--nunchaku_state", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--quant_scope", type=str, default="attn_only",
                        choices=["ffn_only", "attn_only", "dit_full"])
    parser.add_argument("--quant_ffn_blocks", type=str, default=None)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str,
                        choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    return parser.parse_args()


# ---------- hook-based MAC counting (mirrors thop for nunchaku) ----------

_HOOK_RECORDS = []


def _hook_conv(name):
    def fn(m, inp, out):
        x = inp[0]
        g = m.groups
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        k2 = k * k
        o = out
        batch = o.shape[0]
        out_h, out_w = o.shape[2], o.shape[3]
        macs = batch * m.out_channels * (m.in_channels // g) * k2 * out_h * out_w
        _HOOK_RECORDS.append(("conv", name, macs))
    return fn


def _hook_linear(name):
    def fn(m, inp, out):
        x = inp[0]
        flat_batch = 1
        for d in x.shape[:-1]:
            flat_batch *= d
        in_f = m.in_features
        out_f = m.out_features
        _HOOK_RECORDS.append(("linear", name, flat_batch * in_f * out_f))
    return fn


def _hook_svdq(name):
    def fn(m, inp, out):
        x = inp[0]
        flat_batch = 1
        for d in x.shape[:-1]:
            flat_batch *= d
        in_f = m.in_features
        out_f = m.out_features
        macs = flat_batch * in_f * out_f
        rank = getattr(m, "rank", 0)
        if rank > 0:
            macs += flat_batch * rank * (in_f + out_f)
        _HOOK_RECORDS.append(("linear", name, macs))
    return fn


def _register_count_hooks(model):
    handles = []
    for n, m in model.named_modules():
        if isinstance(m, NunchakuSVDQLinear):
            handles.append(m.register_forward_hook(_hook_svdq(n)))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(_hook_linear(n)))
        elif isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_hook_conv(n)))
    return handles


def _count_attn_macs(transformer):
    """Attention: QK^T + AV, both = batch * num_heads * seq_len^2 * head_dim."""
    cfg = transformer.config
    ps = getattr(cfg, "patch_size", 2)
    return 2 * cfg.num_attention_heads * 32 * 32 * cfg.attention_head_dim * cfg.num_layers


# ---------- model class (mirrors test_mac.py TinySR) ----------

class TinySRNunchaku(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

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

        target_suffixes = get_target_suffixes(
            args.quant_scope, ffn_blocks=parse_ffn_blocks(args.quant_ffn_blocks))

        replace_linear_with_nunchaku_w4a4(
            transformer, rank=args.rank,
            target_suffixes=target_suffixes,
            skip_keywords=("lora_",),
        )

        state = load_file(args.nunchaku_state)
        nunchaku_st = {}
        for name, m in transformer.named_modules():
            if isinstance(m, NunchakuSVDQLinear):
                for s in (".qweight", ".wscales", ".proj_down", ".proj_up",
                          ".smooth_factor", ".smooth_factor_orig", ".bias"):
                    k = f"{name}{s}"
                    if k in state:
                        nunchaku_st[k] = state[k]
        transformer.load_state_dict(nunchaku_st, strict=False)

        self.timesteps = torch.tensor(
            [1000.0], device=args.device, dtype=weight_dtype)
        self.vae = vae.to(device=args.device, dtype=weight_dtype)
        self.transformer = transformer.to(args.device, dtype=weight_dtype)

        pooled_prompt_embeds = torch.load(
            os.path.join(args.embedding_dir, "pool_embeds.pt"),
            map_location=args.device).to(dtype=weight_dtype)
        self.pooled_prompt_embeds = pooled_prompt_embeds

    def forward(self, pixel_values, size):
        with torch.no_grad():
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=size, mode='bicubic', align_corners=False)
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to(
                self.args.device, dtype=weight_dtype).clamp(-1, 1)

            model_input = self.vae.encode(
                pixel_values).latents * self.vae.config.scaling_factor
            model_input = model_input.to(self.args.device, dtype=weight_dtype)

            model_pred = self.transformer(
                hidden_states=model_input,
                timestep=self.timesteps,
                pooled_projections=self.pooled_prompt_embeds,
                return_dict=False,
            )[0]
            latent_stu = model_input - model_pred

            image = self.vae.decode(
                latent_stu / self.vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1, 1)
        return image


# ---------- main ----------

if __name__ == "__main__":
    args = parse_args()
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    model = TinySRNunchaku(args)

    input_shape_multi1 = (1, 3, 128, 128)
    num_iterations = args.warmup

    for _ in range(num_iterations):
        dummy_input_multi1 = torch.randn(input_shape_multi1).to(
            device=args.device, dtype=weight_dtype)
        dummy_input_multi2 = (args.process_size, args.process_size)
        model(dummy_input_multi1, dummy_input_multi2)

    if args.device == "cuda":
        torch.cuda.synchronize()

    param_cnt = sum(p.numel() for p in model.parameters())
    print("#Param.", param_cnt / 1e6, "M")

    num_iterations = 1
    print(f"Calculating MACs and Parameters for {num_iterations} iterations...")

    handles = _register_count_hooks(model)
    _HOOK_RECORDS.clear()

    for _ in range(num_iterations):
        dummy_input_multi1 = torch.randn(input_shape_multi1).to(
            device=args.device, dtype=weight_dtype)
        dummy_input_multi2 = (args.process_size, args.process_size)
        model(dummy_input_multi1, dummy_input_multi2)

    if args.device == "cuda":
        torch.cuda.synchronize()

    for h in handles:
        h.remove()

    total_macs = sum(r[2] for r in _HOOK_RECORDS)
    total_macs += _count_attn_macs(model.transformer)
    total_params = sum(p.numel() for p in model.parameters())

    average_macs = total_macs
    average_params = total_params

    average_macs_formatted, average_params_formatted = clever_format(
        [average_macs, average_params], "%.6f")

    print(f"\n--- Average Results over {num_iterations} Iterations ---")
    print(f"Multi-input Model Average MACs: {average_macs_formatted}")
    print(f"Multi-input Model Average Parameters: {average_params_formatted}")
