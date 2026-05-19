"""
Extract patch samples from LR images, matching DIV2K_V2_val naming/format.

Usage examples:

  # LR-only: crop DIV2K_train_LR_x8 into 128x128 patches (stride=64)
  python script/prepare_dataset.py \
    --input_dir dataset/DIV2K_train_LR_x8 \
    --output_dir dataset/DIV2K_train_patches/test_LR \
    --patch_size 128 \
    --stride 64

  # LR+HR pairs
  python script/prepare_dataset.py \
    --input_dir dataset/my_data/LR \
    --hr_dir dataset/my_data/HR \
    --output_dir dataset/my_patches \
    --scale 4 \
    --patch_size 128
"""
import argparse
import os
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract patches from images (avoids stretching).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing LR input images.")
    parser.add_argument("--hr_dir", type=str, default=None,
                        help="Optional: directory containing corresponding HR images. "
                             "If provided, HR patches are also extracted (matching names required).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root directory.")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Patch size in pixels (default: 128).")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window (default: same as patch_size, non-overlapping).")
    parser.add_argument("--scale", type=int, default=None,
                        help="If --hr_dir is given, HR patch size = patch_size * scale. "
                             "If not given, inferred from dir name or filename (e.g. _x4).")
    parser.add_argument("--id_start", type=int, default=1,
                        help="Starting image ID for naming (default: 1 -> 0001_pch_...).")
    parser.add_argument("--image_ext", type=str, default=".png",
                        help="File extension to scan (default: .png).")
    return parser.parse_args()


def natural_sort_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def patch_sliding_window(img, patch_size, stride):
    w, h = img.size
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(img.crop((x, y, x + patch_size, y + patch_size)))
    return patches


def get_image_id(stem, fallback):
    m = re.match(r"(\d+)", stem)
    return int(m.group(1)) if m else fallback


def main():
    args = parse_args()
    if args.stride is None:
        args.stride = args.patch_size

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    lr_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() == args.image_ext.lower()],
        key=natural_sort_key,
    )
    if not lr_files:
        print(f"No *{args.image_ext} files found in {args.input_dir}")
        sys.exit(1)

    # HR setup
    hr_map = {}
    hr_scale = args.scale
    if args.hr_dir:
        hr_dir = Path(args.hr_dir)
        for p in sorted(hr_dir.iterdir(), key=natural_sort_key):
            if p.suffix.lower() == args.image_ext.lower():
                hr_map[p.stem] = p
        if hr_scale is None:
            m = re.search(r"_x(\d+)", str(hr_dir))
            hr_scale = int(m.group(1)) if m else 4
            print(f"  [INFO] inferred HR scale = {hr_scale}x")

    sub_lr = out_dir / "test_LR" if args.hr_dir else out_dir
    os.makedirs(sub_lr, exist_ok=True)
    if args.hr_dir:
        sub_hr = out_dir / "test_HR"
        os.makedirs(sub_hr, exist_ok=True)

    total = 0
    next_id = args.id_start

    for lr_path in tqdm(lr_files, desc="Patch extraction"):
        img_id = get_image_id(lr_path.stem, next_id)
        lr_img = Image.open(lr_path).convert("RGB")

        if lr_img.width < args.patch_size or lr_img.height < args.patch_size:
            lr_patches = [lr_img]
        else:
            lr_patches = patch_sliding_window(lr_img, args.patch_size, args.stride)

        if args.hr_dir:
            hr_img = Image.open(hr_map[lr_path.stem]).convert("RGB")
            hr_ps = args.patch_size * hr_scale
            if hr_img.width < hr_ps or hr_img.height < hr_ps:
                hr_patches = [hr_img]
            else:
                hr_patches = patch_sliding_window(hr_img, hr_ps, args.stride * hr_scale)
            if len(hr_patches) != len(lr_patches):
                print(f"  [WARN] {lr_path.name}: LR patches {len(lr_patches)} != HR patches {len(hr_patches)}; trimming")
                n = min(len(lr_patches), len(hr_patches))
                lr_patches, hr_patches = lr_patches[:n], hr_patches[:n]

        for i, patch_lr in enumerate(lr_patches, start=1):
            name = f"{img_id:04d}_pch_{i:05d}.png"
            patch_lr.save(sub_lr / name)
            total += 1

        if args.hr_dir:
            for i, patch_hr in enumerate(hr_patches, start=1):
                name = f"{img_id:04d}_pch_{i:05d}.png"
                patch_hr.save(sub_hr / name)

        next_id = img_id + 1

    print(f"\nDone. {total} LR patches -> {sub_lr}" +
          (f" + {total} HR patches -> {sub_hr}" if args.hr_dir else ""))


if __name__ == "__main__":
    main()
