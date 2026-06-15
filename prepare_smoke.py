import os, shutil, random

BASE = "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets"
OUT = "/data/disk2/xby/TinySR/dataset/smoke"
os.makedirs(f"{OUT}/vae_stu_lr", exist_ok=True)
os.makedirs(f"{OUT}/vae_stu", exist_ok=True)
os.makedirs(f"{OUT}/latent_stu", exist_ok=True)
os.makedirs(f"{OUT}/pool_embeds", exist_ok=True)

dirs = [
    f"{BASE}/DIV2K/DIV2K_train_LRx4_Real-ESRGAN_Seesr_v2",
    f"{BASE}/FFHQ/FFHQ10K_LRx4_Real-ESRGAN_Seesr_v2",
    f"{BASE}/Flickr2K/Flickr2K_LRx4_Real-ESRGAN_Seesr_v2",
]

all_files = []
for d in dirs:
    for n in os.listdir(f"{d}/vae_stu"):
        all_files.append((d, n))

random.shuffle(all_files)
chosen = all_files[:10]

for d, n in chosen:
    for sub in ["vae_stu", "vae_stu_lr", "latent_stu"]:
        src = f"{d}/{sub}/{n}"
        if os.path.exists(src):
            shutil.copy2(src, f"{OUT}/{sub}/{n}")
    for pool_sub in ["pool_embeds"]:
        src = f"{d}/{pool_sub}/{n}"
        if os.path.exists(src):
            shutil.copy2(src, f"{OUT}/{pool_sub}/{n}")

cnt = len(os.listdir(f"{OUT}/vae_stu_lr"))
print(f"Done: {cnt} samples → {OUT}")
