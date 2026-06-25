"""
Quick collect: run on 4090, save data for outlier visualization.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torchvision import transforms
import json

from models.quant.layers import replace_linear_with_w4a4, set_quant_enabled, set_observer_enabled
from models.quant.components import LowRankBranch
from models.quant.inference import load_models, image_to_latent, get_image_names
from models.quant.tiler import tile_sample

device = torch.device("cuda")
wd = torch.float16

# Load report
with open("outputs/w4a8_svdq_r64_dit_full_a50_20260623_142153/w4a4_report.json") as f:
    report = json.load(f)
alpha_map = {e["name"]: e["weight"]["smooth_alpha"] for e in report["quant_meta"]}

# Load models with LoRA, then merge into base weights
transformer, vae = load_models(
    "checkpoint/tinybackbone/prune-12-merge-tinysr",
    "checkpoint/vae/separable", "checkpoint/tinysr",
    rank=64, cache_dir="/root/autodl-tmp/cache",
    device=device, weight_dtype=wd,
)
transformer = transformer.merge_and_unload()
print("LoRA merged into base weights")
transformer.eval()

ppe = torch.load("dataset/default/pool_embeds.pt", map_location="cpu").to(device=device, dtype=wd)
ts = torch.tensor([1.0], device=device, dtype=wd)

# Replace all Linear with QuantLinear
target_suffixes = [
    "attn.to_q.base_layer", "attn.to_k.base_layer", "attn.to_v.base_layer",
    "attn.to_q", "attn.to_k", "attn.to_v",
    "ff.net.0.proj.base_layer", "ff.net.0.proj",
    "ff.net.2.base_layer", "ff.net.2",
]
replaced = replace_linear_with_w4a4(
    transformer, target_suffixes=target_suffixes,
    weight_quant_kind="svdq",
    weight_quant_kwargs=dict(bits=4, symmetric=True, per_channel=True, ch_axis=0, rank=32, smooth_alpha=0.5, num_svd_iterations=0),
    act_quant_kwargs=dict(bits=8, symmetric=True, per_channel=False),
)

# Enable observer mode
set_quant_enabled(transformer, False)
set_observer_enabled(transformer, True)
for m in transformer.modules():
    if hasattr(m, "weight_quantizer") and hasattr(m.weight_quantizer, "max_gptq_samples"):
        m.weight_quantizer.max_gptq_samples = 999999
        m.weight_quantizer.input_cache = []
        m.weight_quantizer.act_absmax = None

# Find target
TARGET = "transformer_blocks.9.attn.to_k.base_layer"
target = None
for r in replaced:
    if r["name"] == TARGET:
        target = r
        break
assert target is not None, f"Target {TARGET} not found"
target_name = target["name"]

# Run one image
imgs = get_image_names("/root/autodl-tmp/RealSR/LR")
img = imgs[0]
print(f"Image: {img}")

mi, _ = image_to_latent(2, 512, vae, img, transforms.ToTensor(), device, wd)
tile_sample(mi, transformer, timesteps=ts, pooled_prompt_embeds=ppe, weight_dtype=wd, latent_tiled_size=64, latent_tiled_overlap=8)

# Get data
print("Replaced:", [r["name"] for r in replaced if "block" in r["name"] and "9" in r["name"]])
# Get actual module from transformer
target_module = None
for name, mod in transformer.named_modules():
    if "blocks.9" in name and "to_k" in name and hasattr(mod, "weight_quantizer"):
        target_module = mod
        module_path = name
        break
assert target_module is not None, f"Cannot find quantized to_k in block 9. Names: {[n for n,m in transformer.named_modules() if 'blocks.9' in n and 'to_k' in n]}"
print(f"Found at: {module_path}")
wq = target_module.weight_quantizer
inputs = torch.cat(wq.input_cache, dim=0).float()
weight = target_module.weight.data.float()
alpha = alpha_map.get(TARGET, 0.5)
print(f"Act: {inputs.shape}, Weight: {weight.shape}, Alpha: {alpha}")

act_absmax = inputs.abs().max(dim=0).values.cpu()
weight_absmax = weight.abs().max(dim=1).values.cpu()

smooth_scale = (act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)).clamp_min(1e-8)
inputs_smoothed = inputs.cpu() / smooth_scale.unsqueeze(0)
weight_smoothed = weight.cpu() * smooth_scale.unsqueeze(0)

rank_val = 32
branch = LowRankBranch(1536, 1536, rank_val, weight=weight_smoothed.to(wd))
L = branch.get_effective_weight().float().cpu()
residual = (weight_smoothed - L).abs()
svd_err = ((weight_smoothed - L) ** 2).mean().item()
print(f"Act: {inputs.shape}, Weight: {weight.shape}, Alpha: {alpha}")
print(f"SVD err (r={rank_val}): {svd_err:.6e}")

result = dict(
    layer_name=TARGET, alpha=alpha, svd_error=svd_err, img_name=os.path.basename(img),
    act_orig=inputs.abs(), weight_orig=weight.abs(),
    act_smoothed=inputs_smoothed.abs(), weight_smoothed=weight_smoothed.abs(),
    weight_residual=residual,
    smooth_scale=smooth_scale, act_absmax=act_absmax, weight_absmax=weight_absmax,
)
torch.save(result, "/root/autodl-tmp/outlier_data.pt")
print("DONE")
