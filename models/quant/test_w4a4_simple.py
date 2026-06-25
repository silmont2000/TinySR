"""
Simple W4A4 baseline test: LoRA merged, attn-only, no SVD, no smooth.
Run on 4090:
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate tinysr_quant
  cd /root/autodl-tmp/TinySR
  python /root/autodl-tmp/test_w4a4_simple.py
"""
import sys, os, json, time
sys.path.insert(0, "/root/autodl-tmp/TinySR")
import torch, numpy as np
from torchvision import transforms
from PIL import Image
from models.quant.layers import replace_linear_with_w4a4, set_quant_enabled
from models.quant.inference import load_models, image_to_latent, get_image_names
from models.quant.tiler import tile_sample

device = torch.device("cuda"); wd = torch.float16
N_IMG = 10

# Load with LoRA merged
transformer, vae = load_models(
    "checkpoint/tinybackbone/prune-12-merge-tinysr",
    "checkpoint/vae/separable", "checkpoint/tinysr",
    rank=64, cache_dir="/root/autodl-tmp/cache",
    device=device, weight_dtype=wd,
)
transformer = transformer.merge_and_unload()
print("LoRA merged")
transformer.eval()

ppe = torch.load("dataset/default/pool_embeds.pt", map_location="cpu").to(device=device, dtype=wd)
ts = torch.tensor([1.0], device=device, dtype=wd)

# Replace ALL attn layers with plain W4A4 (no SVD)
target_suffixes = [
    "attn.to_q.base_layer", "attn.to_k.base_layer", "attn.to_v.base_layer", "attn.to_out.0.base_layer",
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
]
replaced = replace_linear_with_w4a4(
    transformer, target_suffixes=target_suffixes,
    weight_quant_kind="affine",
    weight_quant_kwargs=dict(bits=4, symmetric=True, per_channel=True, ch_axis=0),
    act_quant_kwargs=dict(bits=4, symmetric=True, per_channel=False),
)
print(f"Replaced {len(replaced)} layers: {[r['name'] for r in replaced[:6]]}...")

# Enable quantization (no observer mode)
set_quant_enabled(transformer, True)

# Run a few images
out_dir = "outputs/w4a8_attnonly_simple_merged"
os.makedirs(out_dir, exist_ok=True)
imgs = get_image_names("/root/autodl-tmp/RealSR/LR")
results = []

for idx in range(min(N_IMG, len(imgs))):
    img = imgs[idx]
    t0 = time.time()
    mi, _ = image_to_latent(2, 512, vae, img, transforms.ToTensor(), device, wd)
    pred = tile_sample(mi, transformer, timesteps=ts, pooled_prompt_embeds=ppe, weight_dtype=wd, latent_tiled_size=64, latent_tiled_overlap=8)
    dt = time.time() - t0
    
    # Decode to image for PSNR
    pred_decoded = vae.decode(pred / vae.config.scaling_factor).sample.detach()
    pred_img = (pred_decoded.squeeze(0).float().clamp(-1, 1) + 1) / 2 * 255
    pred_img = pred_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    # Load ground truth (same filename, different dir: LR -> HR)
    gt_name = os.path.join("/root/autodl-tmp/RealSR/HR", os.path.basename(img))
    gt = np.array(Image.open(gt_name).convert("RGB"))
    
    # PSNR
    mse = np.mean((pred_img.astype(float) - gt.astype(float)) ** 2)
    psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0
    
    # Save output image
    out_path = os.path.join(out_dir, os.path.basename(img))
    Image.fromarray(pred_img).save(out_path)
    
    results.append((os.path.basename(img), psnr, dt))
    print(f"  [{idx+1}/{N_IMG}] {os.path.basename(img)}: PSNR={psnr:.2f} ({dt:.2f}s)")

if results:
    avg_psnr = np.mean([r[1] for r in results])
    print(f"\nAverage PSNR ({len(results)} imgs): {avg_psnr:.2f}")
    with open(f"{out_dir}/psnr.txt", "w") as f:
        for name, psnr, dt in results:
            f.write(f"{name}: PSNR={psnr:.2f} time={dt:.2f}s\n")
        f.write(f"Average: {avg_psnr:.2f}\n")
