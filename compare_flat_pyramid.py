import sys
sys.path.append(".")

import os
import json
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from peft import LoraConfig

from models.vae.autoencoder_tiny import AutoencoderTiny
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.tinysr.pyramid_config import PyramidArchConfig, PStateSpec
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel


NUM_IMAGES = 100
BATCH_SIZE = 1
NUM_STEPS = 200
LR = 1e-4
DEVICE = "cuda:0"
FLICKR2K_HR = "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/Flickr2K/Flickr2K_HR"
VAE_PATH = "checkpoint/vae/separable"
FLAT_CKPT = "checkpoint/tinybackbone/prune-12-merge-tinysr"
POOL_EMBEDS_PATH = "dataset/default/pool_embeds.pt"
OUTPUT_DIR = "outputs/compare_flat_pyramid"

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.RandomCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class SimpleImageDataset(Dataset):
    def __init__(self, hr_dir, num_images, transform=None):
        self.hr_dir = hr_dir
        self.files = sorted(os.listdir(hr_dir))[:num_images]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.hr_dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def make_student_flat():
    model = TinySD3Transformer2DModel.from_pretrained(
        FLAT_CKPT, subfolder="transformer",
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
    )
    model.requires_grad_(False)
    lora_cfg = LoraConfig(
        r=64, lora_alpha=64, init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj", "linear", "linear_1", "linear_2", "net.2",
        ],
    )
    model.add_adapter(lora_cfg, adapter_name="default")
    model.enable_adapters()
    return model


def make_student_pyramid():
    pc = PyramidArchConfig(
        p_states=(
            PStateSpec(num_blocks=4, dim=768, grid_hw=8),
            PStateSpec(num_blocks=4, dim=1152, grid_hw=16),
            PStateSpec(num_blocks=4, dim=1536, grid_hw=32),
        )
    )
    model = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
        FLAT_CKPT, pyramid_config=pc, subfolder="transformer",
        torch_dtype=torch.float16,
    )
    model.requires_grad_(False)
    lora_cfg = LoraConfig(
        r=64, lora_alpha=64, init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj", "linear", "linear_1", "linear_2", "net.2",
        ],
    )
    model.add_adapter(lora_cfg, adapter_name="default")
    model.enable_adapters()
    return model


def train_student(model, dataloader, teacher, vae, pooled_embeds, device, name, num_steps):
    model.to(device=device, dtype=torch.float16)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.train()

    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    losses = []
    t0 = time.time()
    data_iter = iter(dataloader)

    autocast_ctx = torch.autocast("cuda", dtype=torch.float16)

    for step in range(num_steps):
        try:
            lr_img = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            lr_img = next(data_iter)

        lr_img = lr_img.to(device=device, dtype=torch.float16)

        with torch.no_grad():
            latent = vae.encode(lr_img).latents.detach().clone() * vae.config.scaling_factor

        t = torch.tensor([1000], device=device).long()

        with torch.no_grad():
            pred_tea = teacher(
                hidden_states=latent, timestep=t,
                pooled_projections=pooled_embeds.expand(latent.shape[0], -1),
                return_dict=False,
            )[0]
            latent_tea = latent - pred_tea

        with autocast_ctx:
            pred_stu = model(
                hidden_states=latent, timestep=t,
                pooled_projections=pooled_embeds.expand(latent.shape[0], -1),
                return_dict=False,
            )[0]
            latent_stu = latent - pred_stu
            loss = 5 * F.l1_loss(latent_stu.float(), latent_tea.float().detach(), reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg_loss = sum(losses[-20:]) / len(losses[-20:])
            print(f"  [{name}] step {step+1:4d}/{num_steps} | L1={avg_loss:.6f} | ms/step={1000*elapsed/(step+1):.0f}")

    return losses


def main():
    print("=== Loading models ===")

    vae = AutoencoderTiny.from_pretrained(VAE_PATH)
    vae.requires_grad_(False)
    vae.to(DEVICE, dtype=torch.float16)
    print("VAE loaded.")

    teacher = TinySD3Transformer2DModel.from_pretrained(
        FLAT_CKPT, subfolder="transformer",
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
    )
    teacher.requires_grad_(False)
    teacher.to(DEVICE, dtype=torch.float16)
    print("Teacher (flat 12-block) loaded.")

    pooled_embeds = torch.load(POOL_EMBEDS_PATH).to(DEVICE, dtype=torch.float16)
    print(f"Pooled embeds loaded: {pooled_embeds.shape}")

    print(f"\n=== Loading dataset ({NUM_IMAGES} images) ===")
    dataset = SimpleImageDataset(FLICKR2K_HR, NUM_IMAGES, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches.")

    print(f"\n=== Training flat student ({NUM_STEPS} steps) ===")
    flat = make_student_flat()
    flat_losses = train_student(flat, dataloader, teacher, vae, pooled_embeds, DEVICE, "flat", NUM_STEPS)
    del flat
    torch.cuda.empty_cache()

    print(f"\n=== Training pyramid student ({NUM_STEPS} steps) ===")
    pyramid = make_student_pyramid()
    pyramid_losses = train_student(pyramid, dataloader, teacher, vae, pooled_embeds, DEVICE, "pyramid", NUM_STEPS)
    del pyramid
    torch.cuda.empty_cache()

    results = {
        "config": {
            "num_images": NUM_IMAGES,
            "num_steps": NUM_STEPS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        },
        "flat_losses": flat_losses,
        "pyramid_losses": pyramid_losses,
        "flat_final_loss": sum(flat_losses[-20:]) / 20,
        "pyramid_final_loss": sum(pyramid_losses[-20:]) / 20,
    }

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Flat final L1:    {results['flat_final_loss']:.6f}")
    print(f"Pyramid final L1: {results['pyramid_final_loss']:.6f}")


if __name__ == "__main__":
    main()
