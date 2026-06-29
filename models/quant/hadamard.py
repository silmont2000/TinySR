"""Random orthogonal rotation for outlier suppression.

How it works:
  - For each unique in_features size, generate a random orthogonal matrix Q via QR.
  - Q is (n, n), Q @ Q^T = I.  Shared across layers with the same in_features.
  - Weight rotation: W_rot = W @ Q^T  (in-place before SVD + GPTQ)
  - Input rotation:  x_rot = x @ Q^T  (hook during calibration and inference)

No padding required — works for arbitrary in_features sizes (1536, 6144, etc.).
The rotation spreads outliers across all channels, making per-group weight/activation
quantization (64 channels per scale) significantly more accurate.

Storage: Q is saved once per unique size in the nunchaku safetensors.
"""

import torch


def _generate_rotation_matrix(n: int, dtype: torch.dtype = torch.float16,
                              device: torch.device | None = None) -> torch.Tensor:
    """Generate a random (n, n) orthogonal matrix via QR decomposition."""
    R = torch.randn(n, n, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R)
    return Q.to(dtype=dtype).to(device=device)


def _get_or_build_rotation_cache(in_features_set: set[int], dtype, device) -> dict[int, torch.Tensor]:
    """Build a dict of {in_features: rotation_matrix} for all unique sizes."""
    cache = {}
    for n in in_features_set:
        cache[n] = _generate_rotation_matrix(n, dtype=dtype, device=device)
    return cache


def rotate_weight_inplace(weight: torch.Tensor, Q: torch.Tensor) -> None:
    """W ← W @ Q^T (in-place). Q is (n, n) orthogonal."""
    weight.copy_(weight @ Q.T.to(device=weight.device, dtype=weight.dtype))


def make_rotate_input_hook(Q: torch.Tensor, profile: bool = False):
    """Return a pre-forward hook: x ← x @ Q^T."""
    def _hook(module, args):
        x = args[0]
        if profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        x = (x.to(dtype=Q.dtype) @ Q.T.to(device=x.device)).to(dtype=x.dtype)
        if profile:
            end.record()
            torch.cuda.synchronize()
            module._hadamard_time_ms = getattr(module, '_hadamard_time_ms', 0.0) + start.elapsed_time(end)
        return (x, *args[1:])
    return _hook


def enable_rotation(model, seed: int = 0, verbose: bool = True) -> dict[int, torch.Tensor]:
    """Apply random orthogonal rotation to all QuantLinearW4A4 layers.

    Returns a dict of {in_features: Q} for export to safetensors and inference hooks.
    """
    from models.quant.layers import QuantLinearW4A4

    # Collect unique in_features sizes
    sizes = set()
    for _, m in model.named_modules():
        if isinstance(m, QuantLinearW4A4):
            sizes.add(m.in_features)

    dtype = next(m.weight.dtype for _, m in model.named_modules()
                 if isinstance(m, QuantLinearW4A4))
    device = next(m.weight.device for _, m in model.named_modules()
                  if isinstance(m, QuantLinearW4A4))

    rotation_cache = _get_or_build_rotation_cache(sizes, dtype, device)

    count = 0
    for name, m in list(model.named_modules()):
        if not isinstance(m, QuantLinearW4A4):
            continue
        n = m.in_features
        Q = rotation_cache[n]
        rotate_weight_inplace(m.weight.data, Q)
        m.hadamard_signs = n          # store in_features size as marker + key
        m.register_forward_pre_hook(make_rotate_input_hook(Q))
        count += 1

    if verbose:
        print(f"[rotate] applied random orthogonal rotation to {count} layers "
              f"(sizes: {sorted(sizes)})")
    return rotation_cache
