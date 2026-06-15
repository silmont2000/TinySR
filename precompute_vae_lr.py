import sys, os
sys.path.append(".")

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.vae.autoencoder_tiny import AutoencoderTiny
from models.vae.autoencoder_kl  import  AutoencoderKL

DEVICE = "cuda:0"
VAE_PATH = "checkpoint/vae/separable"
DATASETS = [
    ("FFHQ", "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/FFHQ/FFHQ10K_LRx4_Real-ESRGAN_Seesr_v2"),
    ("DIV2K", "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/DIV2K/DIV2K_train_LRx4_Real-ESRGAN_Seesr_v2"),
    ("Flickr2K", "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/Flickr2K/Flickr2K_LRx4_Real-ESRGAN_Seesr_v2"),
]

to_tensor = transforms.ToTensor()


def main():
    # vae = AutoencoderTiny.from_pretrained(VAE_PATH)
    vae = AutoencoderKL.from_pretrained("/data/disk2/xby/sd3-medium", subfolder="vae")
    vae.to(DEVICE, dtype=torch.float16)
    vae.requires_grad_(False)

    total = 0
    for name, base_path in DATASETS:
        lr_dir = os.path.join(base_path, "lr")
        # out_dir = os.path.join(base_path, "vae_stu_lr_256")
        out_dir = os.path.join(base_path, "vae_stu")
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(os.listdir(lr_dir))
        print(f"\n{name}: {len(files)} images from {lr_dir}")

        for fname in tqdm(files, desc=f"  {name}"):
            out_path = os.path.join(out_dir, fname.replace(".png", ".pt"))
            if os.path.exists(out_path):
                continue

            img = Image.open(os.path.join(lr_dir, fname)).convert("RGB")
            img = img.resize((256, 256), Image.BICUBIC)
            tensor = to_tensor(img).unsqueeze(0).to(DEVICE, dtype=torch.float16)
            tensor = tensor * 2 - 1

            with torch.no_grad():
                # latent = vae.encode(tensor).latents * vae.config.scaling_factor
                latent = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor


            torch.save(latent.squeeze(0).cpu(), out_path)
            total += 1

    print(f"\nDone. Total: {total} images processed.")


if __name__ == "__main__":
    main()
