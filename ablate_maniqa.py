#!/usr/bin/env python3
"""逐层 MANIQA 消融 —— 基于 prune-12-merge-tinysr (12 层 fine-tuned 模型)"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import pyiqa
from diffusers import AutoencoderKL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel

CKPT       = os.path.join(SCRIPT_DIR, "checkpoint", "tinybackbone", "prune-12-merge-tinysr")
IMAGE_DIR  = "/data/disk2/xby/TinySR/dataset/StableSR_testsets/DrealSRVal_crop128/test_HR"
OUTPUT_DIR = "/data/disk2/xby/TinySR/outputs/maniqa_ablation"
DEVICE     = "cuda"
TIMESTEP   = 1000
NUM_IMAGES = 10
NUM_LAYERS = 12  # 只有 12 层

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 原始 24 层 → 12 层剪枝后的映射（来自 TinySR prune_mask）
ORIGINAL_MAP = [0, 4, 6, 7, 8, 9, 10, 14, 16, 17, 19, 20]

# ─── 加载模型 ────────────────────────────────────────
print("Loading models...")
vae = AutoencoderKL.from_pretrained("/data/disk2/xby/sd3-medium", subfolder="vae",
                                    torch_dtype=torch.float16).to(DEVICE)
vae.eval()

_cwd = os.getcwd()
os.chdir(SCRIPT_DIR)
try:
    from safetensors.torch import load_file

    with open(os.path.join(CKPT, "transformer", "config.json")) as f:
        cfg = json.load(f)

    transformer = TinySD3Transformer2DModel(
        sample_size=cfg["sample_size"],
        patch_size=cfg["patch_size"],
        in_channels=cfg["in_channels"],
        num_layers=cfg["num_layers"],
        attention_head_dim=cfg["attention_head_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        out_channels=cfg["out_channels"],
        pos_embed_max_size=cfg["pos_embed_max_size"],
        pooled_projection_dim=cfg["pooled_projection_dim"],
    )

    sd = load_file(os.path.join(CKPT, "transformer", "diffusion_pytorch_model.safetensors"))
    pos_embed_ckpt = sd.pop("pos_embed.pos_embed", None)
    missing, unexpected = transformer.load_state_dict(sd, strict=False)
    if pos_embed_ckpt is not None:
        # checkpoint 存的是 192²=36864 token 的 pos_embed，但我们输入 latent 是 64×64
        # → patch_size=2 → 32×32=1024 token，截断到匹配尺寸
        pos_embed_ckpt = pos_embed_ckpt[:, :1024, :]
        transformer.pos_embed._buffers["pos_embed"] = pos_embed_ckpt.to(
            transformer.pos_embed.pos_embed.device)
    print(f"  loaded {len(sd):,} weights, {len(missing)} missing, {len(unexpected)} unexpected")
finally:
    os.chdir(_cwd)

transformer = transformer.to(DEVICE, dtype=torch.float16)
transformer.eval()
transformer.initialized = False  # 阻止预编译，每次 forward 都走正常路径
print(f"  transformer_blocks: {len(transformer.transformer_blocks)}")

# ─── 加载 pooled embeddings ─────────────────────────
pooled_embeds = torch.load(
    "/data/disk2/xby/TinySR/dataset/default/pool_embeds.pt",
    map_location=DEVICE,
).to(torch.float16)
print(f"  pooled_embeds: {pooled_embeds.shape}")

timestep_tensor = torch.tensor([TIMESTEP], device=DEVICE, dtype=torch.float16)

# ─── VAE 预编码测试图 ────────────────────────────────
tensor_transforms = transforms.Compose([transforms.ToTensor()])

image_files = sorted(
    p for p in Path(IMAGE_DIR).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
)[:NUM_IMAGES]
print(f"Test images: {len(image_files)}")

latents_list = []
for p in image_files:
    img = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
    pixel = tensor_transforms(img).unsqueeze(0).to(DEVICE, torch.float16) * 2 - 1
    with torch.no_grad():
        latent = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
    latents_list.append(latent.cpu())

# ─── 手动 forward（不经过 TinySD3 预编译）─────────────
@torch.no_grad()
def run_one(latent_cpu, skip_idx):
    latent = latent_cpu.to(DEVICE)
    hidden_states = transformer.pos_embed(latent)
    temb = transformer.time_text_embed(timestep_tensor, pooled_embeds)

    for i, block in enumerate(transformer.transformer_blocks):
        if i == skip_idx:
            continue
        hidden_states = block(hidden_states=hidden_states, temb=temb)

    hidden_states, _, _ = transformer.norm_out(hidden_states, temb)
    hidden_states = transformer.proj_out(hidden_states)

    # unpatchify
    patch_size = transformer.config.patch_size
    height = latent.shape[-2] // patch_size
    width = latent.shape[-1] // patch_size

    hidden_states = hidden_states.reshape(
        hidden_states.shape[0], height, width, patch_size, patch_size, transformer.out_channels
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        hidden_states.shape[0], transformer.out_channels, height * patch_size, width * patch_size
    )

    denoised = latent - output
    sr = vae.decode(denoised / vae.config.scaling_factor).sample.clamp(-1, 1)
    sr_01 = (sr * 0.5 + 0.5).clamp(0, 1)
    return sr_01.float()


# ─── 加载 MANIQA ─────────────────────────────────────
print("Loading MANIQA...")
maniqa = pyiqa.create_metric("maniqa-pipal", device=DEVICE)

# ─── 逐层消融 ────────────────────────────────────────
results = {}
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\n===== Baseline (no skip) =====", flush=True)
scores_base = []
first_sr = None
for i, lc in enumerate(latents_list):
    sr = run_one(lc, None)
    if i == 0:
        first_sr = sr
    scores_base.append(maniqa(sr).item())
    print(f"  [full] image {i+1}/{len(latents_list)}  MANIQA={scores_base[-1]:.4f}", flush=True)
save_image(first_sr, os.path.join(OUTPUT_DIR, "latest.png"))
score_base = sum(scores_base) / len(scores_base)
results[-1] = score_base
print(f"\n{'full model':>11s}  {score_base:9.4f}  {'—':>9s}")

for skip_idx in range(NUM_LAYERS):
    print(f"\n===== Skip Layer {skip_idx} (orig {ORIGINAL_MAP[skip_idx]}) =====",
          flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scores = []
    for i, lc in enumerate(latents_list):
        sr = run_one(lc, skip_idx)
        scores.append(maniqa(sr).item())
        label = f"skip{skip_idx}"
        print(f"  [{label}] image {i+1}/{len(latents_list)}  MANIQA={scores[-1]:.4f}",
              flush=True)
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

print(f"\n===== MANIQA 重要性排序 =====")
print(f"base MANIQA = {score_base:.4f}")
print(f"{'Rank':>5s}  {'Skip':>5s}  {'Orig':>5s}  {'MANIQA':>8s}  {'Δ':>8s}")
print("-" * 42)
for rank, (layer, score) in enumerate(ranked):
    delta = score - score_base
    print(f"  #{rank+1:2d}   {layer:5d}  {ORIGINAL_MAP[layer]:5d}  {score:8.4f}  {delta:+8.4f}")

# ─── 保存 ────────────────────────────────────────────
out_json = os.path.join(OUTPUT_DIR, "maniqa_ablation.json")
with open(out_json, "w") as f:
    json.dump({
        "base_maniqa": score_base,
        "num_layers": NUM_LAYERS,
        "num_images": NUM_IMAGES,
        "timestep": TIMESTEP,
        "original_mapping": ORIGINAL_MAP,
        "per_layer": results,
        "ranked": [(layer, ORIGINAL_MAP[layer]) for layer, _ in ranked],
    }, f, indent=2)
print(f"\nSaved: {out_json}")
