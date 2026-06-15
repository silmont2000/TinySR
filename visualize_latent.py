import sys, os, argparse
sys.path.append(".")

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from models.vae.autoencoder_tiny import AutoencoderTiny

VAE_PATH = "checkpoint/vae/separable"
DEVICE = "cuda:0"
DTYPE = torch.float16


def decode_latent(latent_path, output_path):
    vae = AutoencoderTiny.from_pretrained(VAE_PATH).to(DEVICE, dtype=DTYPE).eval()

    latent = torch.load(latent_path, map_location=DEVICE).to(DTYPE)
    while latent.dim() > 3 and latent.shape[0] == 1:
        latent = latent.squeeze(0)

    if latent.dim() != 3:
        raise ValueError(f"Expected 3D latent (C,H,W), got {list(latent.shape)}")

    with torch.no_grad():
        decoded = vae.decode(latent.unsqueeze(0) / vae.config.scaling_factor,
                             return_dict=False)[0].clamp(-1, 1)
    img = ToPILImage()((decoded.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu())
    img.save(output_path)
    print(f"Saved: {output_path} ({latent.shape[1]}x{latent.shape[2]} latent → {img.size} image)")


def decode_dir(latent_dir, output_dir, limit=10):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(latent_dir)) if f.endswith(".pt")]
    for f in files[:limit]:
        decode_latent(os.path.join(latent_dir, f), os.path.join(output_dir, f.replace(".pt", ".png")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", help=".pt file or directory of .pt files")
    p.add_argument("--output", "-o", default=None,
                   help="output .png path (for single file) or output dir (for directory)")
    p.add_argument("--limit", type=int, default=10, help="max files from directory")
    args = p.parse_args()

    if os.path.isdir(args.input):
        out_dir = args.output or args.input.replace("/", "_") + "_viz"
        decode_dir(args.input, out_dir, args.limit)
    else:
        out_file = args.output or args.input.replace(".pt", ".png")
        decode_latent(args.input, out_file)
