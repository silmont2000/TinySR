"""DiTAS quantization utilities for TinySR.

DiTAS (Data-free Training-free PTQ for DiTs, WACV 2025) adapted for single-step TinySR:

- Data-free calibration: random pixel tensors -> VAE encode -> random latents.
  Replaces the need for a calibration image dataset.
  TAS (temporal aggregated smoothing) structure is preserved but all samples
  use the same timestep (TinySR is single-step), so max-aggregation collapses
  to max over random latents.

- Asymmetric W4A8: per-channel weight + per-group activation quantization
  with [0, 2^bits-1] range.

- Alternating SVD (10 iters) + per-layer alpha grid search (21 points)
  reuse TinySR's existing calibration infrastructure.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def generate_random_calib_data(vae, timesteps, pooled_prompt_embeds, weight_dtype,
                                device, num_samples=50, pixel_size=512):
    """Generate calibration data from random pixel noise (data-free).

    Random [-1,1] pixel tensors are VAE-encoded to produce latents matching
    TinySR's expected input distribution. This eliminates the need for a
    calibration image dataset while producing latents in the same manifold.

    Args:
        vae: AutoencoderTiny instance.
        timesteps: tensor of shape (1,) or (B,).
        pooled_prompt_embeds: tensor of shape (1, embed_dim).
        weight_dtype: torch.float16 or torch.float32.
        device: torch device.
        num_samples: number of random calibration samples (DiTAS uses 50).
        pixel_size: pixel resolution of random tensors (default 512).

    Returns:
        List of (model_input, timesteps, pooled_prompt_embeds, weight_dtype) tuples.
    """
    calib_data = []
    for _ in range(num_samples):
        pixels = torch.rand(1, 3, pixel_size, pixel_size, device=device, dtype=weight_dtype)
        pixels = pixels * 2 - 1  # [-1, 1] like real VAE inputs
        latent = vae.encode(pixels).latents * vae.config.scaling_factor
        calib_data.append((latent, timesteps, pooled_prompt_embeds, weight_dtype))

    print(f"[DITAS] generated {num_samples} random latent samples "
          f"(pixel {pixel_size}x{pixel_size}, latent {tuple(latent.shape)})")
    return calib_data
