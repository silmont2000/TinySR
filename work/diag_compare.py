import sys
sys.path.append(".")
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.tiler import tile_sample
from utils.util import load_lora_state_dict
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig

wdt = torch.float16
dev = "cuda"
t = TinySD3Transformer2DModel.from_pretrained(
    "checkpoint/tinybackbone/prune-12-merge-tinysr",
    subfolder="transformer", torch_dtype=wdt, low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
)
lc = LoraConfig(
    r=64, lora_alpha=64, init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
)
t.add_adapter(lc)
t.enable_adapters()
sd = StableDiffusion3Pipeline.lora_state_dict("checkpoint/tinysr", weight_name="transformer.safetensors")
load_lora_state_dict(sd, t)
vae = AutoencoderTiny.from_pretrained("checkpoint/vae/separable", torch_dtype=wdt)
vae = vae.to(dev, dtype=wdt).eval()
t = t.to(dev, dtype=wdt).eval()
ts = torch.tensor([1000.0], device=dev, dtype=wdt)
pe = torch.load("dataset/default/pool_embeds.pt", map_location=dev).to(wdt)
tf = transforms.ToTensor()


def run(path, label, tile_vae):
    lr = Image.open(path).convert("RGB")
    W, H = lr.size
    pv = tf(lr).unsqueeze(0).to(dev, dtype=wdt)
    if tile_vae:
        vae.enable_tiling(True)
    with torch.no_grad():
        pv = torch.nn.functional.interpolate(pv, size=(4 * H, 4 * W), mode="bilinear", align_corners=False)
        pv = pv * 2 - 1
        mi = vae.encode(pv).latents * vae.config.scaling_factor
        mi = mi.to(dev, dtype=wdt)
        mp = tile_sample(mi, t, ts, pe, wdt, 64, 8)
        img = vae.decode((mi - mp) / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1, 1)
    r = (img.cpu() / 2 + 0.5).clamp(0, 1).numpy().transpose(1, 2, 0)
    r = (r * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(r).save("/tmp/%s.png" % label)
    g = (np.abs(np.diff(r.astype(np.float32), axis=0)).mean() + np.abs(np.diff(r.astype(np.float32), axis=1)).mean()) / 2
    mi_am = mi.float().abs().mean().item()
    mp_am = mp.float().abs().mean().item()
    print("[%s] tile_vae=%s out %s grad=%.3f mi_am=%.4f mp_am=%.4f mp/mi=%.3f"
          % (label, tile_vae, r.shape, g, mi_am, mp_am, mp_am / max(mi_am, 1e-8)))
    return r


def psnr_vs_upscale(lr_path, sr_arr):
    lr = Image.open(lr_path).convert("RGB")
    up = np.array(lr.resize((sr_arr.shape[1], sr_arr.shape[0]), Image.BILINEAR)).astype(np.float32)
    mse = np.mean((sr_arr.astype(np.float32) - up) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-8))


r1 = run("dataset/div2k_in/0822_pch_00019.png", "div2k_tilevae", True)
r2 = run("dataset/div2k_in/0822_pch_00019.png", "div2k_notile", False)
r3 = run("dataset/diag_crop128/0822_crop128.png", "crop128_notile", False)

print("div2k tile_vae vs notile MAE:", np.abs(r1.astype(np.float32) - r2.astype(np.float32)).mean())
print("div2k notile PSNR vs bilinear-4x: %.2f" % psnr_vs_upscale("dataset/div2k_in/0822_pch_00019.png", r2))
print("crop128 notile PSNR vs bilinear-4x: %.2f" % psnr_vs_upscale("dataset/diag_crop128/0822_crop128.png", r3))
