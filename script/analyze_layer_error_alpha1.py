#!/usr/bin/env python3
"""alpha=1 平滑下逐层量化误差分析（QK 组 / V 单独 / 普通线性层）。

统计口径（每个 block 一组数）：

  QK 组误差（量化 Q、K，V 保持 FP16 参考）：
    * logits_mse      softmax 前 Q_q K_q^T / sqrt(d) 对比 logits_ref
    * prob_mse / kl   softmax 后 attention-probability 对比
    * context_mse     ||(P_q - P_ref) . V_ref||^2

  V 单独误差（固定 P_ref，只量化 V）：
    * context_mse     ||P_ref . (V_q - V_ref)||^2
    * v_out_mse       直接 V 投影输出 MSE（与普通 Linear 对齐）

  out_proj / ff.net.0.proj / ff.net.2 / proj_out：
    * out_mse / out_nmse  直接线性输出误差（无注意力非线性）

量化路径与 _eval_quant_error 完全一致：
    s = act_absmax^alpha / w_absmax^(1-alpha)
    smoothed_weight = W * s，inputs_smoothed = x / s
    SVD 低秩分支 + GPTQ residual（weight_group_size=-1 = per-channel GPTQ）
    activation fake-quant（--act_group_size 64 = per-group A4）
alpha=1.0 时 s 退化为 act_absmax。

实现说明：
    * 模型 FP16 reference forward 在 --device（默认 mps，不可用回退 cpu）上执行，
      pre-hook 只收集各 Linear 的输入（Q/K/V 共享同一个 norm 输出）。
    * --shuffle 时按 --seed 随机采样 --max_images 张图；多张图的 token 拼接后
      统一统计（每张图 token 数相同 => 等权平均），注意力仍按图/tile 边界组合。
    * --max_gptq_samples 默认 2048，与现有 calibration 的 input_cache 上限一致：
      GPTQ 权重只在前 2048 个 token 上拟合，误差统计用全部 token。
    * 离线量化误差在 CPU 上计算（MPS 当前不支持 cholesky，而 GPTQ 需要它）。
    * 直接线性 MSE 与 _eval_quant_error 完全同口径（不含 bias，bias 在误差中抵消）；
      注意力概率/context 组合时对 Q/K/V 两侧都加上原 bias，保持真实工作点。

用法：
    python script/analyze_layer_error_alpha1.py --blocks 0      # 先验证 1 个 block
    python script/analyze_layer_error_alpha1.py --blocks 0-11   # 全部 12 个 block
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.pipeline import get_image_names, image_to_latent, load_models
from models.quant.calibrate import _eval_quant_error, compute_smooth_scale
from models.quant.components import build_quant_component, decompose_svd_branch
from models.quant.inference import build_layer_replacement_kwargs
from models.quant.ops import fake_quant_activation
from models.quant.tiler import tile_sample


def parse_args():
    p = argparse.ArgumentParser(description="alpha=1 逐层量化误差分析")
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    p.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    p.add_argument("--lora_dir", type=str, default="checkpoint/tinysr")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--embedding_dir", type=str, default="dataset/default/")
    p.add_argument("--input_dir", type=str, default="../testset/RealSR/LR")
    p.add_argument("--output_dir", type=str, default="outputs/alpha1_layer_error")
    p.add_argument("--max_images", type=int, default=1)
    p.add_argument("--shuffle", action="store_true",
                   help="从输入目录随机采样 --max_images 张图（用 --seed 固定）")
    p.add_argument("--max_gptq_samples", type=int, default=2048,
                   help="GPTQ 使用的最大 token 数（与 calibration input_cache 上限一致；0=不限）")
    p.add_argument("--blocks", type=str, default="all",
                   help="Block indices: '0', '0,2,5', '0-11', or 'all'")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--w_bits", type=int, default=4)
    p.add_argument("--a_bits", type=int, default=4)
    p.add_argument("--svdq_rank", type=int, default=32)
    p.add_argument("--act_group_size", type=int, default=64)
    p.add_argument("--weight_group_size", type=int, default=-1,
                   help="-1 = per-channel GPTQ（与 _eval_quant_error 默认一致）")

    p.add_argument("--device", type=str, default="mps",
                   help="模型 FP16 forward 设备（mps 不可用时自动回退 cpu）")
    p.add_argument("--offline_device", type=str, default="cpu",
                   help="离线量化误差计算设备（MPS 不支持 cholesky，默认 cpu）")
    p.add_argument("--attn_device", type=str, default=None,
                   help="注意力组合（logits/softmax/context）计算设备，默认跟随 --device")
    p.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--process_size", type=int, default=512)
    p.add_argument("--latent_tiled_size", type=int, default=64)
    p.add_argument("--latent_tiled_overlap", type=int, default=8)
    p.add_argument("--timestep", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_device(name):
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("mps", "cuda"):
        print(f"[device] {name} 不可用，回退到 cpu")
    return torch.device("cpu")


def parse_blocks(spec, n_blocks):
    if spec == "all":
        return list(range(n_blocks))
    blocks = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            blocks.extend(range(int(a), int(b) + 1))
        else:
            blocks.append(int(part))
    blocks = sorted(set(blocks))
    invalid = [b for b in blocks if not 0 <= b < n_blocks]
    if invalid:
        raise ValueError(f"block 越界: {invalid} (模型共 {n_blocks} 个 block)")
    return blocks


class Capture:
    """收集某 Linear 输入：保留 chunk（保持 tile 边界）并累计 per-channel absmax。"""

    def __init__(self, num_channels, dtype):
        self.chunks = []
        self.act_absmax = torch.zeros(num_channels, dtype=dtype)

    def update(self, x):
        x = x.detach()
        self.chunks.append(x.cpu())
        am = x.abs().reshape(-1, x.shape[-1]).amax(dim=0).cpu()
        torch.maximum(self.act_absmax, am, out=self.act_absmax)

    def concat(self):
        """拼接所有 chunk -> (N_total, C)，保留原输入 dtype（fp16）。"""
        return torch.cat(self.chunks, dim=0).reshape(-1, self.act_absmax.shape[0])

    def chunk_sizes(self):
        return [c.shape[0] * c.shape[1] for c in self.chunks]

    def chunk_batch(self):
        return [c.shape[0] for c in self.chunks]


@torch.no_grad()
def quantized_linear_output(weight, act_absmax, weight_absmax, wq, inputs, alpha,
                            act_bits=4, act_symmetric=True, act_scale=None,
                            act_group_size=64, num_iterations=0, output_device=None,
                            gptq_inputs=None,
                            no_gptq=False, no_svd_early_stop=False):
    """与 _eval_quant_error 完全相同的离线量化路径，但返回 (ref_out, q_out)。

    返回的均为 Linear 无 bias 输出（bias 在误差中抵消）；调用方如需真实工作点
    （注意力组合）可自行加 bias。SVD/GPTQ/激活量化在 inputs 所在设备（CPU）上
    执行（MPS 不支持 cholesky）；输出矩阵乘法在 output_device（默认 MPS）上执行，
    避免 CPU fp16 matmul 成为瓶颈。
    """
    if output_device is None:
        output_device = inputs.device
    smooth_scale = compute_smooth_scale(act_absmax, weight_absmax, alpha)
    smoothed_weight = weight * smooth_scale.reshape(1, -1)
    inputs_smoothed = inputs / smooth_scale.reshape(1, -1)
    if gptq_inputs is None:
        gptq_inputs = inputs
    inputs_smoothed_gptq = gptq_inputs / smooth_scale.reshape(1, -1)
    orig_out = inputs.to(output_device) @ weight.to(output_device).T

    branch, residual_q, _ = decompose_svd_branch(
        smoothed_weight,
        rank=wq.rank,
        alpha=wq.alpha,
        bits=wq.quantizer.bits,
        symmetric=wq.quantizer.symmetric,
        eps=wq.quantizer.eps,
        inputs=inputs_smoothed_gptq,
        gptq_block_size=wq.gptq_block_size,
        gptq_damp_percentage=wq.gptq_damp_percentage,
        weight_group_size=getattr(wq, "weight_group_size", -1),
        num_iterations=num_iterations,
        no_gptq=no_gptq,
        no_svd_early_stop=no_svd_early_stop,
    )
    q_inputs = fake_quant_activation(
        inputs_smoothed, bits=act_bits, symmetric=act_symmetric,
        eps=wq.quantizer.eps, scale=act_scale, group_size=act_group_size,
    )
    branch_out = branch(inputs_smoothed) if branch is not None else 0.0
    q_out = q_inputs.to(output_device) @ residual_q.to(output_device).T
    if isinstance(branch_out, torch.Tensor):
        q_out = q_out + branch_out.to(output_device)
    return orig_out, q_out


def linear_metrics(ref, q):
    """返回 (mse, nmse, n)。nmse = sum(err^2) / sum(ref^2)。"""
    ref = ref.float()
    err = (ref - q.float())
    sumsq_err = err.pow(2).sum().item()
    sumsq_ref = ref.pow(2).sum().item()
    n = ref.numel()
    mse = sumsq_err / max(n, 1)
    nmse = sumsq_err / max(sumsq_ref, 1e-30)
    return mse, nmse, n


def _split_heads(t, batch, seq, heads, head_dim):
    return t.view(batch, seq, heads, head_dim).transpose(1, 2)


@torch.no_grad()
def qk_group_metrics(q_ref, q_q, k_ref, k_q, v_ref, v_q,
                     chunk_sizes, chunk_batches, heads, head_dim,
                     device=torch.device("cpu")):
    """逐 tile（保持注意力边界）组合 Q/K/V，统计 QK 组与 V 的注意力口径误差。"""
    scale = head_dim ** -0.5
    logits_se = 0.0
    logits_sref = 0.0
    logits_n = 0
    prob_se = 0.0
    prob_sref = 0.0
    prob_n = 0
    kl_sum = 0.0
    n_queries = 0
    qk_ctx_se = 0.0
    qk_ctx_sref = 0.0
    qk_ctx_n = 0
    v_ctx_se = 0.0
    v_ctx_sref = 0.0
    v_ctx_n = 0

    q_ref_chunks = q_ref.to(device).split(chunk_sizes)
    q_q_chunks = q_q.to(device).split(chunk_sizes)
    k_ref_chunks = k_ref.to(device).split(chunk_sizes)
    k_q_chunks = k_q.to(device).split(chunk_sizes)
    v_ref_chunks = v_ref.to(device).split(chunk_sizes)
    v_q_chunks = v_q.to(device).split(chunk_sizes)

    for i, n_tok in enumerate(chunk_sizes):
        batch = chunk_batches[i]
        qr = _split_heads(q_ref_chunks[i].float(), batch, n_tok, heads, head_dim)
        qq = _split_heads(q_q_chunks[i].float(), batch, n_tok, heads, head_dim)
        kr = _split_heads(k_ref_chunks[i].float(), batch, n_tok, heads, head_dim)
        kq = _split_heads(k_q_chunks[i].float(), batch, n_tok, heads, head_dim)
        vr = _split_heads(v_ref_chunks[i].float(), batch, n_tok, heads, head_dim)
        vq = _split_heads(v_q_chunks[i].float(), batch, n_tok, heads, head_dim)

        logits_ref = torch.matmul(qr, kr.transpose(-1, -2)) * scale
        logits_qk = torch.matmul(qq, kq.transpose(-1, -2)) * scale
        p_ref = torch.softmax(logits_ref, dim=-1)
        p_qk = torch.softmax(logits_qk, dim=-1)

        logits_err = logits_qk - logits_ref
        logits_se += logits_err.pow(2).sum().item()
        logits_sref += logits_ref.pow(2).sum().item()
        logits_n += logits_ref.numel()

        prob_err = p_qk - p_ref
        prob_se += prob_err.pow(2).sum().item()
        prob_sref += p_ref.pow(2).sum().item()
        prob_n += p_ref.numel()

        p_ref_c = p_ref.clamp_min(1e-12)
        p_qk_c = p_qk.clamp_min(1e-12)
        kl_sum += (p_ref_c * (p_ref_c.log() - p_qk_c.log())).sum(dim=-1).sum().item()
        n_queries += p_ref.shape[0] * p_ref.shape[1] * p_ref.shape[2]

        context_ref = torch.matmul(p_ref, vr)
        qk_ctx_err = torch.matmul(prob_err, vr)
        qk_ctx_se += qk_ctx_err.pow(2).sum().item()
        qk_ctx_sref += context_ref.pow(2).sum().item()
        qk_ctx_n += qk_ctx_err.numel()

        v_ctx_err = torch.matmul(p_ref, vq - vr)
        v_ctx_se += v_ctx_err.pow(2).sum().item()
        v_ctx_sref += context_ref.pow(2).sum().item()
        v_ctx_n += v_ctx_err.numel()

    return {
        "logits_mse": logits_se / max(logits_n, 1),
        "logits_nmse": logits_se / max(logits_sref, 1e-30),
        "prob_mse": prob_se / max(prob_n, 1),
        "prob_nmse": prob_se / max(prob_sref, 1e-30),
        "prob_kl": kl_sum / max(n_queries, 1),
        "qk_context_mse": qk_ctx_se / max(qk_ctx_n, 1),
        "qk_context_nmse": qk_ctx_se / max(qk_ctx_sref, 1e-30),
        "v_context_mse": v_ctx_se / max(v_ctx_n, 1),
        "v_context_nmse": v_ctx_se / max(v_ctx_sref, 1e-30),
        "n_logits": logits_n,
        "n_prob": prob_n,
        "n_queries": n_queries,
        "n_qk_ctx": qk_ctx_n,
        "n_v_ctx": v_ctx_n,
    }


def build_wq(args):
    kw = build_layer_replacement_kwargs(
        args.w_bits, args.a_bits, args.svdq_rank, args.alpha,
        svdq_iterations=0, act_group_size=args.act_group_size,
        weight_group_size=args.weight_group_size,
    )
    return build_quant_component(**kw["weight_quant_kwargs"])


def quant_one_linear(linear, capture, args, offline_device, output_device,
                     gptq_cap=0, verify=False):
    """对单个 Linear 做离线量化，返回 (ref, q, act_absmax)。"""
    weight = linear.weight.detach().to(device=offline_device, dtype=linear.weight.dtype)
    # 现有 _eval_quant_error 的 clamp_min(1e-8) 在 fp16 下会把 1e-8 下溢成 0：
    # 单张图出现全零输入通道时 smooth_scale=0 -> x/0 -> NaN。
    # 这里 clamp 到 fp16 可表示的最小正值，其余路径完全不变。
    eps = torch.finfo(weight.dtype).tiny if weight.dtype == torch.float16 else 1e-8
    act_absmax = capture.act_absmax.to(device=offline_device, dtype=weight.dtype).clamp_min(eps)
    weight_absmax = weight.detach().abs().amax(dim=0).clamp_min(eps)
    inputs = capture.concat().to(device=offline_device, dtype=weight.dtype)
    gptq_inputs = inputs[:gptq_cap] if gptq_cap and gptq_cap > 0 else inputs
    wq = build_wq(args)

    ref, q = quantized_linear_output(
        weight, act_absmax, weight_absmax, wq, inputs, args.alpha,
        act_bits=args.a_bits, act_symmetric=True, act_scale=None,
        act_group_size=args.act_group_size, output_device=output_device,
        gptq_inputs=gptq_inputs,
    )

    if verify:
        ref_v, q_v = quantized_linear_output(
            weight, act_absmax, weight_absmax, wq, gptq_inputs, args.alpha,
            act_bits=args.a_bits, act_symmetric=True, act_scale=None,
            act_group_size=args.act_group_size, output_device=output_device,
        )
        err = _eval_quant_error(
            weight, act_absmax, weight_absmax, wq, gptq_inputs, args.alpha,
            act_bits=args.a_bits, act_symmetric=True, act_scale=None,
            act_group_size=args.act_group_size,
        ).item()
        helper_err = (ref_v.float() - q_v.float()).pow(2).mean().item()
        print(f"[verify] helper={helper_err:.6e} _eval_quant_error={err:.6e} "
              f"rel={abs(helper_err - err) / max(abs(err), 1e-30):.3e}")
    return ref, q


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    offline_device = torch.device(args.offline_device)
    attn_device = resolve_device(args.attn_device) if args.attn_device else device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/6] 加载 backbone + LoRA (device={device}, dtype={weight_dtype})...")
    t0 = time.time()
    transformer, vae = load_models(
        args.pretrained_model_name_or_path, args.vae_path, args.lora_dir,
        args.lora_rank, args.cache_dir, device, weight_dtype,
    )
    transformer = transformer.merge_and_unload()
    transformer = transformer.eval()
    print(f"      模型加载完成 ({time.time() - t0:.1f}s)")

    n_blocks = len(transformer.transformer_blocks)
    blocks = parse_blocks(args.blocks, n_blocks)
    print(f"[2/6] block 选择: {blocks} (共 {n_blocks})")

    # ---- 注册 pre-hook 收集输入 --------------------------------------------
    captures = {}
    handles = []

    def register(key, module_name):
        module = transformer.get_submodule(module_name)
        cap = Capture(module.in_features, weight_dtype)
        captures[key] = cap

        def _hook(_mod, inp, _cap=cap):
            _cap.update(inp[0])

        handles.append(module.register_forward_pre_hook(_hook))

    for b in blocks:
        prefix = f"transformer_blocks.{b}"
        register(f"{prefix}.attn.in", f"{prefix}.attn.to_q")       # Q/K/V 共享输入
        register(f"{prefix}.attn.ctx", f"{prefix}.attn.to_out.0")  # out_proj 输入
        register(f"{prefix}.ff.up", f"{prefix}.ff.net.0.proj")
        register(f"{prefix}.ff.down", f"{prefix}.ff.net.2")
    register("proj_out", "proj_out")

    # ---- FP16 reference forward -------------------------------------------
    image_names = get_image_names(args.input_dir)
    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        image_names = rng.sample(image_names, min(args.max_images, len(image_names)))
    else:
        image_names = image_names[:args.max_images]
    if not image_names:
        raise SystemExit(f"没有找到输入图片: {args.input_dir}")
    print(f"[3/6] FP16 reference forward: {len(image_names)} 张图 <- {args.input_dir}"
          + ("（随机采样）" if args.shuffle else ""))
    for n in image_names:
        print(f"      {os.path.basename(n)}")

    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=device,
    ).to(dtype=weight_dtype)
    timesteps = torch.tensor([args.timestep], device=device, dtype=weight_dtype)
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    t0 = time.time()
    with torch.no_grad():
        for image_path in tqdm(image_names, desc="[forward]"):
            latent, _ = image_to_latent(
                args.upscale, args.process_size, vae, image_path,
                tensor_transform, device, weight_dtype,
            )
            tile_sample(
                latent, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap,
            )
    for h in handles:
        h.remove()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"      forward 完成 ({time.time() - t0:.1f}s)")

    # ---- 离线逐层量化误差 ------------------------------------------------
    print(f"[4/6] 离线量化误差 (alpha={args.alpha}, w{args.w_bits}a{args.a_bits}, "
          f"rank={args.svdq_rank}, act_gs={args.act_group_size}, "
          f"weight_gs={args.weight_group_size}, device={offline_device})...")
    t0 = time.time()

    rows = []
    per_block = {}
    verify_done = False

    for b in blocks:
        prefix = f"transformer_blocks.{b}"
        attn = transformer.get_submodule(f"{prefix}.attn")
        heads = attn.heads
        head_dim = attn.to_q.out_features // heads

        # Q / K / V（共享同一输入，各自独立量化路径）
        cap_in = captures[f"{prefix}.attn.in"]
        chunk_sizes = cap_in.chunk_sizes()
        chunk_batches = cap_in.chunk_batch()

        q_lin = transformer.get_submodule(f"{prefix}.attn.to_q")
        k_lin = transformer.get_submodule(f"{prefix}.attn.to_k")
        v_lin = transformer.get_submodule(f"{prefix}.attn.to_v")

        t_layer = time.time()
        q_ref, q_q = quant_one_linear(q_lin, cap_in, args, offline_device,
                                      attn_device, args.max_gptq_samples,
                                      verify=not verify_done)
        verify_done = True
        k_ref, k_q = quant_one_linear(k_lin, cap_in, args, offline_device,
                                      attn_device, args.max_gptq_samples)
        v_ref, v_q = quant_one_linear(v_lin, cap_in, args, offline_device,
                                      attn_device, args.max_gptq_samples)
        t_qkv = time.time()

        # 注意力组合使用真实工作点（两侧都加 bias）
        b_q = q_lin.bias.detach().to(device=q_ref.device, dtype=q_ref.dtype)
        b_k = k_lin.bias.detach().to(device=k_ref.device, dtype=k_ref.dtype)
        b_v = v_lin.bias.detach().to(device=v_ref.device, dtype=v_ref.dtype)
        attn_qk = qk_group_metrics(
            q_ref + b_q.reshape(1, -1), q_q + b_q.reshape(1, -1),
            k_ref + b_k.reshape(1, -1), k_q + b_k.reshape(1, -1),
            v_ref + b_v.reshape(1, -1), v_q + b_v.reshape(1, -1),
            chunk_sizes, chunk_batches, heads, head_dim, device=attn_device,
        )
        t_attn = time.time()

        v_out_mse, v_out_nmse, v_out_n = linear_metrics(v_ref, v_q)

        # out_proj
        ctx_lin = transformer.get_submodule(f"{prefix}.attn.to_out.0")
        ctx_ref, ctx_q = quant_one_linear(ctx_lin, captures[f"{prefix}.attn.ctx"],
                                          args, offline_device, attn_device,
                                          args.max_gptq_samples)
        out_proj_mse, out_proj_nmse, out_proj_n = linear_metrics(ctx_ref, ctx_q)

        # FFN up / down
        up_lin = transformer.get_submodule(f"{prefix}.ff.net.0.proj")
        up_ref, up_q = quant_one_linear(up_lin, captures[f"{prefix}.ff.up"],
                                        args, offline_device, attn_device,
                                        args.max_gptq_samples)
        up_mse, up_nmse, up_n = linear_metrics(up_ref, up_q)

        down_lin = transformer.get_submodule(f"{prefix}.ff.net.2")
        down_ref, down_q = quant_one_linear(down_lin, captures[f"{prefix}.ff.down"],
                                             args, offline_device, attn_device,
                                             args.max_gptq_samples)
        down_mse, down_nmse, down_n = linear_metrics(down_ref, down_q)
        t_block = time.time()
        print(f"      block {b}: qkv_quant={t_qkv - t_layer:.1f}s "
              f"attn_compose={t_attn - t_qkv:.1f}s linear_quant={t_block - t_attn:.1f}s")

        block_rows = [
            {"block": b, "layer": f"{prefix}.attn.qk", "metric": "logits_mse",
             "mse": attn_qk["logits_mse"], "nmse": attn_qk["logits_nmse"],
             "n_elements": attn_qk["n_logits"]},
            {"block": b, "layer": f"{prefix}.attn.qk", "metric": "prob_mse",
             "mse": attn_qk["prob_mse"], "nmse": attn_qk["prob_nmse"],
             "n_elements": attn_qk["n_prob"]},
            {"block": b, "layer": f"{prefix}.attn.qk", "metric": "prob_kl",
             "mse": attn_qk["prob_kl"], "nmse": None,
             "n_elements": attn_qk["n_queries"]},
            {"block": b, "layer": f"{prefix}.attn.qk", "metric": "context_mse",
             "mse": attn_qk["qk_context_mse"], "nmse": attn_qk["qk_context_nmse"],
             "n_elements": attn_qk["n_qk_ctx"]},
            {"block": b, "layer": f"{prefix}.attn.v", "metric": "context_mse",
             "mse": attn_qk["v_context_mse"], "nmse": attn_qk["v_context_nmse"],
             "n_elements": attn_qk["n_v_ctx"]},
            {"block": b, "layer": f"{prefix}.attn.v", "metric": "out_mse",
             "mse": v_out_mse, "nmse": v_out_nmse, "n_elements": v_out_n},
            {"block": b, "layer": f"{prefix}.attn.out_proj", "metric": "out_mse",
             "mse": out_proj_mse, "nmse": out_proj_nmse, "n_elements": out_proj_n},
            {"block": b, "layer": f"{prefix}.ff.up", "metric": "out_mse",
             "mse": up_mse, "nmse": up_nmse, "n_elements": up_n},
            {"block": b, "layer": f"{prefix}.ff.down", "metric": "out_mse",
             "mse": down_mse, "nmse": down_nmse, "n_elements": down_n},
        ]
        rows.extend(block_rows)
        per_block[b] = {
            "qk_logits": (attn_qk["logits_mse"], attn_qk["logits_nmse"]),
            "qk_prob": (attn_qk["prob_mse"], attn_qk["prob_nmse"]),
            "qk_kl": attn_qk["prob_kl"],
            "qk_ctx": (attn_qk["qk_context_mse"], attn_qk["qk_context_nmse"]),
            "v_ctx": (attn_qk["v_context_mse"], attn_qk["v_context_nmse"]),
            "v_out": (v_out_mse, v_out_nmse),
            "out_proj": (out_proj_mse, out_proj_nmse),
            "ff_up": (up_mse, up_nmse),
            "ff_down": (down_mse, down_nmse),
        }

    # proj_out（全局一层）
    po_lin = transformer.get_submodule("proj_out")
    po_ref, po_q = quant_one_linear(po_lin, captures["proj_out"], args,
                                    offline_device, attn_device,
                                    args.max_gptq_samples)
    po_mse, po_nmse, po_n = linear_metrics(po_ref, po_q)
    rows.append({"block": "", "layer": "proj_out", "metric": "out_mse",
                 "mse": po_mse, "nmse": po_nmse, "n_elements": po_n})
    print(f"      离线量化完成 ({time.time() - t0:.1f}s)")

    # ---- 输出 CSV -----------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "layer_error_alpha1.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block", "layer", "metric", "mse", "nmse", "n_elements"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "block": r["block"],
                "layer": r["layer"],
                "metric": r["metric"],
                "mse": f"{r['mse']:.8e}",
                "nmse": "" if r["nmse"] is None else f"{r['nmse']:.8e}",
                "n_elements": r["n_elements"],
            })
    print(f"[5/6] CSV -> {csv_path} ({len(rows)} 行)")

    # ---- 控制台汇总 ---------------------------------------------------------
    print("[6/6] 按 block 汇总（mse (nmse)，KL 无 nmse）")
    header = (f"{'block':>5} {'qk_logits':>18} {'qk_prob':>18} {'qk_kl':>10} "
              f"{'qk_ctx':>18} {'v_ctx':>18} {'v_out':>18} {'out_proj':>18} "
              f"{'ff_up':>18} {'ff_down':>18} {'proj_out':>18}")
    print(header)
    print("-" * len(header))

    def cell(pair):
        mse, nmse = pair
        return f"{mse:.2e} ({nmse:.3f})"

    for b in blocks:
        d = per_block[b]
        print(f"{b:>5} {cell(d['qk_logits']):>18} {cell(d['qk_prob']):>18} "
              f"{d['qk_kl']:>10.3e} {cell(d['qk_ctx']):>18} {cell(d['v_ctx']):>18} "
              f"{cell(d['v_out']):>18} {cell(d['out_proj']):>18} "
              f"{cell(d['ff_up']):>18} {cell(d['ff_down']):>18} {cell((po_mse, po_nmse)):>18}")
    print()
    print(f"proj_out: out_mse={po_mse:.6e} out_nmse={po_nmse:.6f} n={po_n}")


if __name__ == "__main__":
    main()
