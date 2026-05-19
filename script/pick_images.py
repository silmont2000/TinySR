import os
import shutil
import random
import sys
from pathlib import Path

src = Path("/data/disk2/xby/TinySR/dataset/DIV2K_train_patches/test_LR")
dst = Path("/data/disk2/xby/TinySR/dataset/DIV2K_train_patches/pick")

if not src.exists():
    print(f"[ERROR] source not found: {src}")
    sys.exit(1)

all_files = [p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
all_files.sort()
print(f"Source: {len(all_files)} images in {src}")

if len(all_files) < 50:
    print(f"Only {len(all_files)} available, picking all of them.")
    picked = all_files
else:
    picked = random.sample(all_files, 50)

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

ok = 0
for f in picked:
    shutil.copy2(str(f), str(dst / f.name))
    ok += 1

actual = len(list(dst.iterdir()))
print(f"Copied {ok} images -> {dst}  (verified: {actual} files in target)")
