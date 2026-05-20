from models.quant.components import LowRankBranch
from models.quant.layers import (
    set_quant_enabled,
    set_observer_enabled,
    QuantLinearW4A4,
)
from models.quant.inference import (
    get_weight_dtype,
    load_models,
    replace_quant_layers,
)
from thop import profile, clever_format
import torch.nn as nn
import torch
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="W4A4 quantized MACs benchmark.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str,
                        default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str,
                        default="checkpoint/tinysr")
    parser.add_argument("--cache_dir", type=str,
                        default="/data/disk2/xby/models")
    parser.add_argument("--embedding_dir", type=str,
                        default="dataset/default/")

    parser.add_argument("--load_quant_state", type=str, required=True,
                        help="Path to quantized state_dict (.pt)")
    parser.add_argument("--quant_scope", type=str,
                        choices=["ffn_only", "attn_only", "dit_full"],
                        default="dit_full")
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--svdq_rank", type=int, default=32)
    parser.add_argument("--svdq_smooth_alpha", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str,
                        choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--rank", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--timestep", type=float, default=1000.0)

    return parser.parse_args()


class QuantizedTinySR(nn.Module):
    def __init__(self, transformer, vae, prompt_embeds, pooled_prompt_embeds,
                 timesteps, device, weight_dtype):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.timesteps = timesteps
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.device = device
        self.weight_dtype = weight_dtype

    def forward(self, pixel_values, size):
        with torch.no_grad():
            pixel_values = nn.functional.interpolate(
                pixel_values, size=size,
                mode='bicubic', align_corners=False)
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to(
                self.device, dtype=self.weight_dtype).clamp(-1, 1)

            model_input = self.vae.encode(
                pixel_values).latents * self.vae.config.scaling_factor
            model_input = model_input.to(
                self.device, dtype=self.weight_dtype)

            model_pred = self.transformer(
                hidden_states=model_input,
                timestep=self.timesteps,
                pooled_projections=self.pooled_prompt_embeds,
                return_dict=False,
            )[0]
            latent_stu = model_input - model_pred

            image = self.vae.decode(
                latent_stu / self.vae.config.scaling_factor,
                return_dict=False,
            )[0].squeeze(0).clamp(-1, 1)
        return image


def count_quant_linear_w4a4(m, x, y):
    output_elements = y[0].numel() if isinstance(y, tuple) else y.numel()
    total_ops = output_elements * m.in_features
    wq = m.weight_quantizer
    branch = getattr(wq, 'branch', None)
    if branch is not None and hasattr(branch, 'rank') and branch.rank > 0:
        batch = output_elements // m.out_features
        total_ops += batch * (
            m.in_features * branch.rank + branch.rank * m.out_features)
    m.total_ops += torch.DoubleTensor([total_ops]).to(m.total_ops.device)


if __name__ == "__main__":
    args = parse_args()
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.mixed_precision)

    print(
        f"[mac] device={args.device}  dtype={args.mixed_precision}  scope={args.quant_scope}")

    # 1. Load base model
    print("[mac] loading model ...")
    transformer, vae = load_models(
        args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
        args.rank, args.cache_dir, device, weight_dtype, skip_lora=False)

    # 2. Replace Linear -> QuantLinearW4A4
    print(f"[mac] replacing Linear -> QuantLinearW4A4 "
          f"(w{args.w_bits}a{args.a_bits} r{args.svdq_rank}) ...")
    replaced = replace_quant_layers(
        transformer, args.quant_scope, None,
        args.w_bits, args.a_bits,
        args.svdq_rank, args.svdq_smooth_alpha)
    print(f"[mac]   {len(replaced)} layers replaced")

    # 3. Load quantized state dict (same logic as test_quant_timing.py)
    print(f"[mac] loading state_dict <- {args.load_quant_state}")
    state = torch.load(args.load_quant_state, map_location="cpu")
    model_sd = transformer.state_dict()

    matched = {k: v for k, v in state.items()
               if k in model_sd and v.shape == model_sd[k].shape}
    missing, unexpected = transformer.load_state_dict(matched, strict=False)

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
        found = False
        for pname, param in target.named_parameters(recurse=False):
            if pname == attr_name:
                param.data = v
                reshaped += 1
                found = True
                break
        if not found:
            for bname, buf in target.named_buffers(recurse=False):
                if bname == attr_name:
                    target._buffers[attr_name] = v
                    reshaped += 1
                    break

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
            wq.residual = state[r_key].to(
                device=device_v, dtype=dtype_v)
            quant_injected += 1

        s_key = prefix + "smooth_scale"
        if s_key in state and state[s_key] is not None:
            wq.smooth_scale = state[s_key].to(
                device=device_v, dtype=dtype_v)

        a_key = prefix + "act_absmax"
        if a_key in state and state[a_key] is not None:
            wq.act_absmax = state[a_key].to(
                device=device_v, dtype=dtype_v)

        b_a_key = prefix + "branch.a.weight"
        b_b_key = prefix + "branch.b.weight"
        if b_a_key in state and b_b_key in state:
            in_f = m.in_features
            out_f = m.out_features
            rank = wq.rank
            if rank > 0:
                branch = LowRankBranch(
                    in_f, out_f, rank=rank, alpha=wq.alpha, weight=None)
                branch.to(device=device_v, dtype=dtype_v)
                branch.a.weight.data.copy_(
                    state[b_a_key].to(device=device_v, dtype=dtype_v))
                branch.b.weight.data.copy_(
                    state[b_b_key].to(device=device_v, dtype=dtype_v))
                wq.branch = branch

    not_found_skipped = sum(
        1 for k in state
        if k not in model_sd and ".weight_quantizer." in k)

    if reshaped:
        print(f"[mac]   shape-changed (direct assign): {reshaped}")
    if not_found - not_found_skipped:
        print(f"[mac]   not in model (skipped): "
              f"{not_found - not_found_skipped}")
    if missing:
        print(f"[mac]   missing keys: {len(missing)}")
    if unexpected:
        print(f"[mac]   unexpected keys: {len(unexpected)}")
    if quant_injected:
        print(f"[mac]   quantizer state injected: "
              f"{quant_injected} layers")

    set_observer_enabled(transformer, False)
    set_quant_enabled(transformer, True)

    # 4. Load prompt embeddings & timesteps
    prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "prompt_embeds.pt"),
        map_location=device).to(dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"),
        map_location=device).to(dtype=weight_dtype)
    timesteps = torch.tensor(
        [args.timestep], device=device, dtype=weight_dtype)

    # 5. Build wrapped model (VAE + quant transformer in one forward)
    model = QuantizedTinySR(
        transformer, vae, prompt_embeds, pooled_prompt_embeds,
        timesteps, device, weight_dtype)

    # 6. Parameter count
    param_cnt = sum(p.numel() for p in model.parameters())
    print(f"\n#Param. {param_cnt/1e6:.2f}M")

    # 7. Profile MACs (same dummy shapes as test_mac.py)
    input_shape = (1, 3, 128, 128)
    dummy_input = torch.randn(input_shape).to(
        device=device, dtype=weight_dtype)
    dummy_size = (512, 512)

    print(f"\n[mac] profiling with input {input_shape}, "
          f"target size {dummy_size} ...")

    custom_ops = {QuantLinearW4A4: count_quant_linear_w4a4}
    macs, params = profile(
        model, inputs=(dummy_input, dummy_size),
        custom_ops=custom_ops, verbose=False)

    macs_fmt, params_fmt = clever_format([macs, params], "%.6f")
    print(f"\n--- Quantized MACs ({args.quant_scope}, "
          f"w{args.w_bits}a{args.a_bits}) ---")
    print(f"MACs     : {macs_fmt}  ({macs:.2e})")
    print(f"Params   : {params_fmt}  ({params:.2e})")
    print(f"#Param.  : {param_cnt/1e6:.2f}M")
