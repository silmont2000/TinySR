#!/usr/bin/env python3
"""逐层 MANIQA 消融 —— 基于完整 SD3 (24 层)"""

import os
import sys
import json
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import pyiqa
from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler

# ─── 配置 ───────────────────────────────────────────
CKPT       = "/data/disk2/xby/sd3-medium"
IMAGE_DIR  = "/data/disk2/xby/TinySR/dataset/StableSR_testsets/DrealSRVal_crop128/test_HR"
OUTPUT_DIR = "/data/disk2/xby/TinySR/outputs/maniqa_ablation_sd3"
DEVICE     = "cuda"
TIMESTEP   = 1000
NUM_IMAGES = 10
NUM_LAYERS = 24
NUM_STEPS  = 10  # 去噪步数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 加载模型 ────────────────────────────────────────
print("Loading models...")
vae = AutoencoderKL.from_pretrained(CKPT, subfolder="vae",
                                    torch_dtype=torch.float16).to(DEVICE)
vae.eval()

transformer = SD3Transformer2DModel.from_pretrained(
    CKPT, subfolder="transformer", torch_dtype=torch.float16,
).to(DEVICE)
transformer.eval()
print(f"  transformer_blocks: {len(transformer.transformer_blocks)}")

# ─── 加载调度器 ─────────────────────────────────────
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    CKPT, subfolder="scheduler")
scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

# ─── 加载 prompt embeddings ─────────────────────────
prompt_dir = "/data/disk2/xby/TinySR/dataset/default"
prompt_embeds = torch.load(
    os.path.join(prompt_dir, "prompt_embeds.pt"),
    map_location=DEVICE,
).to(torch.float16)
pooled_embeds = torch.load(
    os.path.join(prompt_dir, "pool_embeds.pt"),
    map_location=DEVICE,
).to(torch.float16)
print(f"  prompt_embeds: {prompt_embeds.shape}  pooled_embeds: {pooled_embeds.shape}")

# ─── VAE 预编码测试图 ────────────────────────────────
tensor_transforms = transforms.Compose([transforms.ToTensor()])

image_files = sorted(
    p for p in Path(IMAGE_DIR).iterdir()
    if p.suffix.lower() in (".png", ".jpg", ".jpeg")
)[:NUM_IMAGES]
print(f"Test images: {len(image_files)}")

latents_list = []
for p in image_files:
    img = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
    pixel = tensor_transforms(img).unsqueeze(0).to(DEVICE, torch.float16) * 2 - 1
    with torch.no_grad():
        latent = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
    latents_list.append(latent.cpu())

# ─── 手动 forward ───────────────────────────────────
@torch.no_grad()
def run_one(latent_cpu, skip_idx):
    latent = latent_cpu.to(DEVICE)
    noise = torch.randn_like(latent)
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)  # 每张图重置 step_index
    latent_noisy = latent * (1 - scheduler.sigmas[0]) + noise * scheduler.sigmas[0]

    for t in scheduler.timesteps:
        t_tensor = t.unsqueeze(0)
        hidden_states = transformer.pos_embed(latent_noisy)
        temb = transformer.time_text_embed(t_tensor, pooled_embeds)
        encoder_hidden_states = transformer.context_embedder(prompt_embeds)

        for i, block in enumerate(transformer.transformer_blocks):
            if i == skip_idx:
                continue
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        hidden_states = transformer.norm_out(hidden_states, temb)
        hidden_states = transformer.proj_out(hidden_states)

        # unpatchify
        ps = transformer.config.patch_size
        h = latent.shape[-2] // ps
        w = latent.shape[-1] // ps
        model_output = hidden_states.reshape(
            hidden_states.shape[0], h, w, ps, ps, transformer.config.out_channels,
        )
        model_output = torch.einsum("nhwpqc->nchpwq", model_output)
        model_output = model_output.reshape(
            hidden_states.shape[0], transformer.config.out_channels, h * ps, w * ps,
        )

        # 调度器步进
        latent_noisy = scheduler.step(model_output, t, latent_noisy).prev_sample

    # 解码最终结果
    sr = vae.decode(latent_noisy / vae.config.scaling_factor).sample.clamp(-1, 1)
    sr_01 = (sr * 0.5 + 0.5).clamp(0, 1)
    return sr_01.float()


# ─── 加载 MANIQA ─────────────────────────────────────
print("Loading MANIQA...")
maniqa = pyiqa.create_metric("maniqa-pipal", device=DEVICE)

# ─── 逐层消融 ────────────────────────────────────────
results = {}
torch.cuda.empty_cache()

print("\n===== Baseline (no skip) =====", flush=True)
scores_base = []
first_sr = None
for i, lc in enumerate(latents_list):
    sr = run_one(lc, None)
    if i == 0:
        first_sr = sr
    scores_base.append(maniqa(sr).item())
    print(f"  [full] image {i+1}/{len(latents_list)}  MANIQA={scores_base[-1]:.4f}",
          flush=True)
save_image(first_sr, os.path.join(OUTPUT_DIR, "latest.png"))
score_base = sum(scores_base) / len(scores_base)
results[-1] = score_base
print(f"\n{'full model':>11s}  {score_base:9.4f}  {'---':>9s}")

for skip_idx in range(NUM_LAYERS):
    print(f"\n===== Skip Layer {skip_idx} =====", flush=True)
    torch.cuda.empty_cache()
    scores = []
    for i, lc in enumerate(latents_list):
        sr = run_one(lc, skip_idx)
        scores.append(maniqa(sr).item())
        print(f"  [skip{skip_idx:02d}] image {i+1}/{len(latents_list)}"
              f"  MANIQA={scores[-1]:.4f}", flush=True)
    first_sr = run_one(latents_list[0], skip_idx)
    save_image(first_sr, os.path.join(OUTPUT_DIR, "latest.png"))
    score = sum(scores) / len(scores)
    delta = score - score_base
    results[skip_idx] = score
    print(f"    Block {skip_idx:2d}   {score:9.4f}  {delta:+9.4f}")

# ─── 排序输出 ────────────────────────────────────────
ranked = sorted(
    [(layer, score) for layer, score in results.items() if layer >= 0],
    key=lambda x: x[1],
)

print(f"\n===== MANIQA 重要性排序 (24 层) =====")
print(f"base MANIQA = {score_base:.4f}")
print(f"\n{'Rank':>5s}  {'Layer':>6s}  {'MANIQA':>8s}  {'Delta':>8s}")
print("-" * 35)
for rank, (layer, score) in enumerate(ranked):
    delta = score - score_base
    print(f"  #{rank+1:2d}    {layer:5d}   {score:8.4f}  {delta:+8.4f}")

# ─── 保存 ────────────────────────────────────────────
out_json = os.path.join(OUTPUT_DIR, "maniqa_ablation.json")
with open(out_json, "w") as f:
    json.dump({
        "base_maniqa": score_base,
        "num_layers": NUM_LAYERS,
        "num_images": NUM_IMAGES,
        "timestep": TIMESTEP,
        "per_layer": results,
        "ranked": [layer for layer, _ in ranked],
    }, f, indent=2)
print(f"\nSaved: {out_json}")
