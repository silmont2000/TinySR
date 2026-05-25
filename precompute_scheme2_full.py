import sys, os
sys.path.append(".")

import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm

from models.tinysr.tinysd3 import TinySD3Transformer2DModel

CKPT = "checkpoint/tinybackbone/prune-12-merge-tinysr"
GRIDS = [8, 16]
BLOCKS_PER_SNAPSHOT = 4
OUT_SUBDIRS = {8: "latent_stu_teacher_8", 16: "latent_stu_teacher_16"}

DATASETS = [
    "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/DIV2K/DIV2K_train_LRx4_Real-ESRGAN_Seesr_v2",
    "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/Flickr2K/Flickr2K_LRx4_Real-ESRGAN_Seesr_v2",
    "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/FFHQ/FFHQ10K_LRx4_Real-ESRGAN_Seesr_v2",
]


def pool_tokens(tokens, grid_in, grid_out):
    B, N, C = tokens.shape
    factor = grid_in // grid_out
    t = tokens.reshape(B, grid_in, grid_in, C).permute(0, 3, 1, 2)
    t = F.avg_pool2d(t, kernel_size=factor, stride=factor)
    return t.permute(0, 2, 3, 1).flatten(1, 2)


def unpatchify(noise, grid_hw, patch_size, out_channels):
    B = noise.shape[0]
    o = noise.reshape(B, grid_hw, grid_hw, patch_size, patch_size, out_channels)
    o = torch.einsum("nhwpqc->nchpwq", o)
    return o.reshape(B, out_channels, grid_hw * patch_size, grid_hw * patch_size)


def worker(gpu_id, jobs):
    device = f"cuda:{gpu_id}"
    dtype = torch.float16

    teacher = TinySD3Transformer2DModel.from_pretrained(
        CKPT, subfolder="transformer", low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, torch_dtype=dtype,
    ).to(device).eval()

    for base_path, fname in tqdm(jobs, desc=f"GPU{gpu_id}", position=gpu_id):
        vae_path = os.path.join(base_path, "vae_stu", fname)
        pool_path = os.path.join(base_path, "pool_embeds", fname)

        if not os.path.exists(pool_path):
            continue

        all_exist = all(os.path.exists(os.path.join(base_path, OUT_SUBDIRS[hw], fname)) for hw in GRIDS)
        if all_exist:
            continue

        vae = torch.load(vae_path, map_location=device).to(dtype)
        if vae.dim() == 3:
            vae = vae.unsqueeze(0)

        pooled = torch.load(pool_path, map_location=device).to(dtype)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        t_tensor = torch.tensor([1000], device=device).long()

        with torch.no_grad():
            h = teacher.pos_embed(vae)
            temb = teacher.time_text_embed(t_tensor, pooled)

            source_grid = vae.shape[-1] // teacher.pos_embed.proj.stride[0]
            patch_size = teacher.pos_embed.proj.stride[0]
            out_channels = teacher.proj_out.out_features // (patch_size * patch_size)

            targets = {}
            for block_idx, block in enumerate(teacher.transformer_blocks):
                if block.initialized:
                    h = block.forward_(hidden_states=h)
                else:
                    h = block(hidden_states=h, temb=temb)

                if (block_idx + 1) % BLOCKS_PER_SNAPSHOT == 0:
                    stage_idx = block_idx // BLOCKS_PER_SNAPSHOT
                    if stage_idx >= len(GRIDS):
                        continue
                    target_hw = GRIDS[stage_idx]
                    tokens_pooled = pool_tokens(h, source_grid, target_hw)
                    noise = teacher.proj_out(tokens_pooled)
                    noise_latent = unpatchify(noise, target_hw, patch_size, out_channels)
                    input_pooled = F.avg_pool2d(vae, kernel_size=source_grid // target_hw)
                    refined = input_pooled - noise_latent
                    targets[target_hw] = refined.squeeze(0).cpu().half()

            for hw, tgt in targets.items():
                torch.save(tgt, os.path.join(base_path, OUT_SUBDIRS[hw], fname))

    del teacher
    torch.cuda.empty_cache()


def main():
    gpu_ids = [1, 2, 3, 5, 6]
    print(f"Using GPUs: {gpu_ids}")

    # Create target directories
    for base_path in DATASETS:
        for hw in GRIDS:
            d = os.path.join(base_path, OUT_SUBDIRS[hw])
            os.makedirs(d, exist_ok=True)
            print(f"  mkdir {d}")
    print()

    # Collect all jobs
    all_jobs = []
    for base_path in DATASETS:
        vae_dir = os.path.join(base_path, "vae_stu")
        if not os.path.exists(vae_dir):
            continue
        for fname in sorted(os.listdir(vae_dir)):
            # Check if any target is missing
            need = any(not os.path.exists(os.path.join(base_path, OUT_SUBDIRS[hw], fname)) for hw in GRIDS)
            if need:
                all_jobs.append((base_path, fname))

    print(f"Total jobs: {len(all_jobs)}")

    # Split across GPUs
    n_gpus = len(gpu_ids)
    chunks = [[] for _ in range(n_gpus)]
    for i, job in enumerate(all_jobs):
        chunks[i % n_gpus].append(job)

    set_start_method("spawn", force=True)
    procs = []
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        p = Process(target=worker, args=(gpu_id, chunks[gpu_idx]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All done.")


if __name__ == "__main__":
    main()
