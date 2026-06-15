import sys, os
sys.path.append(".")
import torch, traceback
from PIL import Image
from torchvision import transforms
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny

d = "cuda:0"
dtype = torch.float16
teacher = TinySD3Transformer2DModel.from_pretrained(
    "checkpoint/tinybackbone/prune-12-merge-tinysr",
    subfolder="transformer", low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True, torch_dtype=dtype,
).to(d).eval()
teacher.pos_embed.pos_embed = teacher.pos_embed.pos_embed[:, :64, :].contiguous().clone()

vae = AutoencoderTiny.from_pretrained("checkpoint/vae/separable").to(d, dtype).eval()
img_path = "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/FFHQ/FFHQ10K_LRx4_Real-ESRGAN_Seesr_v2/lr/0010001.png"
img = Image.open(img_path).convert("RGB")
tensor = transforms.ToTensor()(img).unsqueeze(0).to(d, dtype) * 2 - 1
with torch.no_grad():
    latent = vae.encode(tensor).latents * vae.config.scaling_factor
    pooled = torch.load("dataset/default/pool_embeds.pt", map_location=d).to(dtype)
    t = torch.tensor([1000], device=d).long()
    print(f"latent: {list(latent.shape)}, range [{latent.min():.2f}, {latent.max():.2f}]")

    pred = teacher(hidden_states=latent, timestep=t, pooled_projections=pooled, return_dict=True).sample
    print(f"pred: {list(pred.shape)}")
    result = latent - pred
    print(f"result: {list(result.shape)}, range [{result.min():.2f}, {result.max():.2f}]")
    
    out = vae.decode(result / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
    out_img = transforms.ToPILImage()((out.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu().float())
    os.makedirs("outputs", exist_ok=True)
    out_img.save("outputs/teacher_128_test.png")
    print("saved outputs/teacher_128_test.png — left=input(128×128), right=teacher(128×128) output")
