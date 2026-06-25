import sys, os
sys.path.append(".")

import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm

from models.tinysr.tinysd3 import TinySD3Transformer2DModel

CKPT = "checkpoint/tinybackbone/prune-12-merge-tinysr"
BLOCK_TO_GRID = {
    11: 32,
}
GRIDS = list(BLOCK_TO_GRID.values())
OUT_SUBDIRS = {8: "latent_stu_teacher_8", 16: "latent_stu_teacher_16", 32: "latent_stu_teacher_32"}
QKV_SUBDIR = "latent_stu_teacher_32_qkv_12"

DATASETS = [
 "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/DIV2K/DIV2K_train_LRx4_Real-ESRGAN_Seesr_v2",
  "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/Flickr2K/Flickr2K_LRx4_Real-ESRGAN_Seesr_v2",
  "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/FFHQ/FFHQ10K_LRx4_Real-ESRGAN_Seesr_v2",
]


def worker(gpu_id, jobs):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
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
        qkv_exist = os.path.exists(os.path.join(base_path, QKV_SUBDIR, fname))
        if all_exist and qkv_exist:
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

            targets = {}
            target_qkv = {}
            for block_idx, block in enumerate(teacher.transformer_blocks):
                need_qkv = block_idx in BLOCK_TO_GRID
                if block.initialized:
                    if need_qkv:
                        h, q, k, v = block.forward_(hidden_states=h, return_qkv=True)
                    else:
                        h = block.forward_(hidden_states=h)
                else:
                    if need_qkv:
                        h, q, k, v = block(hidden_states=h, temb=temb, return_qkv=True)
                    else:
                        h = block(hidden_states=h, temb=temb)

                if block_idx in BLOCK_TO_GRID:
                    target_hw = BLOCK_TO_GRID[block_idx]
                    assert h.shape[0] == 1, f"precompute expects batch_size=1, got {h.shape[0]}"
                    targets[target_hw] = h.squeeze(0).cpu().half()
                    target_qkv[target_hw] = {
                        "q": q.squeeze(0).cpu().half(),
                        "k": k.squeeze(0).cpu().half(),
                        "v": v.squeeze(0).cpu().half(),
                    }
                    if len(BLOCK_TO_GRID) == len(targets):
                        break

            for hw, tgt in targets.items():
                torch.save(tgt, os.path.join(base_path, OUT_SUBDIRS[hw], fname))
            for hw, qkv in target_qkv.items():
                torch.save(qkv, os.path.join(base_path, QKV_SUBDIR, fname))

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    gpu_ids = [1,7]
    print(f"Using GPUs: {gpu_ids}")

    for base_path in DATASETS:
        for hw in GRIDS:
            d = os.path.join(base_path, OUT_SUBDIRS[hw])
            os.makedirs(d, exist_ok=True)
            print(f"  mkdir {d}")
        d = os.path.join(base_path, QKV_SUBDIR)
        os.makedirs(d, exist_ok=True)
        print(f"  mkdir {d}")
    print()

    all_jobs = []
    for base_path in DATASETS:
        vae_dir = os.path.join(base_path, "vae_stu")
        if not os.path.exists(vae_dir):
            continue
        for fname in sorted(os.listdir(vae_dir)):
            need = any(not os.path.exists(os.path.join(base_path, OUT_SUBDIRS[hw], fname)) for hw in GRIDS) \
                or not os.path.exists(os.path.join(base_path, QKV_SUBDIR, fname))
            if need:
                all_jobs.append((base_path, fname))

    print(f"Total jobs: {len(all_jobs)}")

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
