import sys, os
sys.path.append(".")

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.tinysr.tinysd3 import TinySD3Transformer2DModel

SMOKE = "dataset/smoke"
OUT = "dataset/smoke_scheme2"
DEVICE = "cuda:0"
CKPT = "checkpoint/tinybackbone/prune-12-merge-tinysr"
GRIDS = [8, 16, 32]
BLOCKS_PER_SNAPSHOT = 4


def pool_tokens(tokens, grid_hw_in, grid_hw_out):
    B, N, C = tokens.shape
    factor = grid_hw_in // grid_hw_out
    t = tokens.reshape(B, grid_hw_in, grid_hw_in, C).permute(0, 3, 1, 2)
    t = F.avg_pool2d(t, kernel_size=factor, stride=factor)
    return t.permute(0, 2, 3, 1).flatten(1, 2)


def unpatchify(noise, grid_hw, patch_size, out_channels):
    B = noise.shape[0]
    o = noise.reshape(B, grid_hw, grid_hw, patch_size, patch_size, out_channels)
    o = torch.einsum("nhwpqc->nchpwq", o)
    return o.reshape(B, out_channels, grid_hw * patch_size, grid_hw * patch_size)


def extract_teacher_targets(fname):
    teacher = TinySD3Transformer2DModel.from_pretrained(
        CKPT, subfolder="transformer", low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    vae = torch.load(os.path.join(SMOKE, "vae_stu", fname), map_location=DEVICE).to(torch.float16)
    if vae.dim() == 3:
        vae = vae.unsqueeze(0)

    pooled = torch.load(os.path.join(SMOKE, "pool_embeds", fname), map_location=DEVICE).to(torch.float16)
    if pooled.dim() == 1:
        pooled = pooled.unsqueeze(0)
    t = torch.tensor([1000], device=DEVICE).long()

    h = teacher.pos_embed(vae)
    temb = teacher.time_text_embed(t, pooled)
    del teacher.time_text_embed
    teacher.temb = temb

    source_grid = vae.shape[-1] // teacher.pos_embed.proj.stride[0]  # 32 for 512px
    targets = {}

    for block_idx, block in enumerate(teacher.transformer_blocks):
        if block_idx == 0 and not teacher.initialized:
            h = block(hidden_states=h, temb=teacher.temb)
        elif block.initialized:
            h = block.forward_(hidden_states=h)
        else:
            h = block(hidden_states=h, temb=teacher.temb)

        if (block_idx + 1) % BLOCKS_PER_SNAPSHOT == 0:
            stage_idx = block_idx // BLOCKS_PER_SNAPSHOT
            target_hw = GRIDS[stage_idx]

            tokens_pooled = pool_tokens(h, source_grid, target_hw)
            noise = teacher.proj_out(tokens_pooled)
            oc = teacher.proj_out.out_features // (teacher.pos_embed.proj.stride[0] ** 2)
            noise_latent = unpatchify(noise, target_hw, teacher.pos_embed.proj.stride[0], oc)

            input_pooled = F.avg_pool2d(vae, kernel_size=source_grid // target_hw)
            refined = input_pooled - noise_latent
            targets[target_hw] = refined.squeeze(0).cpu().half()

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return targets


def main():
    for g in GRIDS:
        os.makedirs(os.path.join(OUT, f"target_{g}"), exist_ok=True)
    files = sorted(os.listdir(os.path.join(SMOKE, "latent_stu")))
    for fname in tqdm(files):
        targets = extract_teacher_targets(fname)
        for hw, tgt in targets.items():
            torch.save(tgt, os.path.join(OUT, f"target_{hw}", fname))
    print(f"Done. {len(files)} images → {OUT}")


if __name__ == "__main__":
    main()
