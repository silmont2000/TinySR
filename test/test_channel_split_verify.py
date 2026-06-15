#!/usr/bin/env python3
"""
Verify channel-split hypothesis: for each block individually,
compute H/L channel indices from that block's INPUT spatial sensitivity,
apply split (H keep 32², L downsample to 16²), measure output error vs baseline.
"""
import sys, os
_TINYSR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TINYSR_ROOT)

import torch, numpy as np, glob
import torch.nn.functional as F
from PIL import Image; from torchvision import transforms
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny


def set_seed(seed=42):
    import random; random.seed(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spatial_err_per_channel(X):
    """X: (C, 32, 32) — per-channel 32→16→32 reconstruction error."""
    ds = F.interpolate(X.unsqueeze(0), size=(16,16), mode='bilinear')
    us = F.interpolate(ds, size=(32,32), mode='bilinear').squeeze(0)
    return (X - us).abs().mean(dim=(1,2)) / (X.abs().mean(dim=(1,2)) + 1e-10)


def compute_indices(h_32_map, budget=0.40):
    """h_32_map: (1536, 32, 32) — spatial feature map of block INPUT.
    Returns (h_idx, l_idx) tensors on same device."""
    err = spatial_err_per_channel(h_32_map).cpu().numpy()
    sorted_idx = np.argsort(err)[::-1]
    h_ch = int((budget * 1536 - 384) / 0.75)
    h_ch = max(1, min(1535, h_ch))
    h_idx = torch.tensor(sorted_idx[:h_ch].copy(), device=h_32_map.device, dtype=torch.long)
    l_idx = torch.tensor(sorted_idx[h_ch:].copy(), device=h_32_map.device, dtype=torch.long)
    return h_idx, l_idx


def test_block(block, h_in):
    """
    h_in: (B, 1024, 1536) — block INPUT.
    Tests multiple budgets, returns list of (budget, error%).
    """
    B, N, D = h_in.shape
    X = h_in[0].reshape(32, 32, D).permute(2, 0, 1)  # (1536, 32, 32)

    with torch.no_grad():
        h_baseline = block.forward_(hidden_states=h_in)
    h_ref_32 = h_baseline.abs().float().mean() + 1e-8  # for rel error

    results = []
    for budget in [0.35, 0.40, 0.45, 0.50, 0.60]:
        h_idx, l_idx = compute_indices(X, budget)

        h_c = h_in[:, :, h_idx]
        l_c = h_in[:, :, l_idx]
        B, N, l_ch = l_c.shape

        l_map = l_c.reshape(B, 32, 32, l_ch).permute(0, 3, 1, 2)
        l_down = F.interpolate(l_map.float(), size=(16, 16), mode='bilinear').half()
        l_up = F.interpolate(l_down.float(), size=(32, 32), mode='bilinear').half()
        l_flat = l_up.permute(0, 2, 3, 1).reshape(B, N, l_ch)

        h_merged = torch.zeros_like(h_in)
        h_merged[:, :, h_idx] = h_c
        h_merged[:, :, l_idx] = l_flat

        with torch.no_grad():
            h_split = block.forward_(hidden_states=h_merged)

        err = float((h_split.float() - h_baseline.float()).abs().mean()) / float(h_ref_32)
        h_ch_val = len(h_idx)
        results.append((budget, h_ch_val, err))
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--blocks", type=int, default=6)
    args = p.parse_args()
    set_seed(42)

    print("[1/3] Loading models...")
    m = TinySD3Transformer2DModel.from_pretrained(
        'checkpoint/tinybackbone/prune-12-merge-tinysr', subfolder='transformer',
        torch_dtype=torch.float16, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, local_files_only=True).cuda()
    m.eval()
    vae = AutoencoderTiny.from_pretrained('checkpoint/vae/separable',
        torch_dtype=torch.float16, local_files_only=True).cuda().eval()
    pool = torch.load('dataset/default/pool_embeds.pt', map_location='cuda', weights_only=True)[:1].half()
    t = torch.tensor([1000.], device='cuda', dtype=torch.float16)
    tt = transforms.Compose([transforms.ToTensor()])

    with torch.no_grad():
        _ = m(hidden_states=torch.randn(1, 16, 64, 64, device='cuda', dtype=torch.float16),
              timestep=t, pooled_projections=pool, return_dict=False)

    print(f"[2/3] Running block-level tests on {args.num_images} images...")
    n_blks = args.blocks
    budget_list = [0.35, 0.40, 0.45, 0.50, 0.60]
    per_blk = {bi: {b: [] for b in budget_list} for bi in range(n_blks)}

    imgs = sorted(glob.glob('dataset/StableSR_testsets/DrealSRVal_crop128/test_LR/*.png'))
    test_imgs = imgs[:args.num_images]

    for idx, img_path in enumerate(test_imgs):
        img = Image.open(img_path).convert('RGB').resize((512, 512), Image.BICUBIC)
        px = tt(img).unsqueeze(0).cuda().half() * 2 - 1
        with torch.no_grad():
            lat = vae.encode(px).latents * vae.config.scaling_factor
            lat = F.interpolate(lat.float(), size=(64, 64), mode='bilinear').half()
            h = m.pos_embed(lat)
            for bi in range(n_blks):
                for budget, h_ch_val, err in test_block(m.transformer_blocks[bi], h):
                    per_blk[bi][budget].append(err)
                h = m.transformer_blocks[bi].forward_(hidden_states=h)
        if (idx + 1) % 5 == 0:
            print(f"  {idx+1}/{len(test_imgs)}")
        torch.cuda.empty_cache()

    print(f"\n[3/3] Results (block-level relative L1 error, avg over {len(test_imgs)} images):")
    print(f"  Budget = fraction of full 32² compute")
    print()
    header = f"  {'Blk':>4}"
    for b in budget_list:
        header += f"  {'budget='+str(int(b*100))+'%':>15}"
    print(header)
    dash = f"  {'':>4}"
    for b in budget_list:
        dash += f"  {'H_ch   err%':>15}"
    print(dash)
    print(f"  {'-'*(4 + 15*len(budget_list))}")

    for bi in range(n_blks):
        row = f"   {bi:>4}"
        for b in budget_list:
            errs = per_blk[bi][b]
            if errs:
                avg = np.mean(errs) * 100
                # H count depends only on budget, not block
                h_ch = int((b * 1536 - 384) / 0.75)
                h_ch = max(1, min(1535, h_ch))
                row += f"  {h_ch:>4}ch {avg:>6.2f}%"
            else:
                row += f"  {'--':>15}"
        print(row)

    print()
    for b in budget_list:
        avg_all = np.mean([np.mean(per_blk[bi][b]) for bi in range(n_blks) if per_blk[bi][b]]) * 100
        print(f"  budget={int(b*100):>3}%  avg block err = {avg_all:.2f}%")


if __name__ == "__main__":
    main()
