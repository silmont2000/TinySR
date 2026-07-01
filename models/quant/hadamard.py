"""Random orthogonal rotation for outlier suppression.

Uses dense QR matrix, no padding, O(n²) per forward pass.
Works for any in_features size. 1536 stays 1536, 6144 stays 6144.

Applies an orthogonal rotation to both weights (calibration) and inputs
(inference hook), cancelling out to preserve the original linear layer output.
"""

import math

import torch
from tqdm import tqdm


# ── Shared utilities ────────────────────────────────────────────────────

def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ── Mode: random_orthogonal (dense QR, no padding) ─────────────────────

def _generate_rotation_matrix(n: int, dtype: torch.dtype = torch.float16,
                              device: torch.device | None = None) -> torch.Tensor:
    R = torch.randn(n, n, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R)
    return Q.to(dtype=dtype).to(device=device)


def _build_random_orthogonal_cache(sizes: set[int], dtype, device) -> dict[int, torch.Tensor]:
    cache = {}
    for n in sizes:
        cache[n] = _generate_rotation_matrix(n, dtype=dtype, device=device)
    return cache


def _rotate_weight_dense(weight: torch.Tensor, Q: torch.Tensor) -> None:
    weight.copy_(weight @ Q.T.to(device=weight.device, dtype=weight.dtype))


def _make_dense_rotation_hook(Q: torch.Tensor):
    def _hook(module, args):
        x = args[0]
        x = (x.to(dtype=Q.dtype) @ Q.T.to(device=x.device)).to(dtype=x.dtype)
        return (x, *args[1:])
    return _hook


# ── Unified entry point ─────────────────────────────────────────────────

def enable_rotation(model, mode: str = "random_orthogonal", seed: int = 0,
                    verbose: bool = True) -> dict:
    """Apply random orthogonal rotation to all QuantLinearW4A4 layers.

    Parameters
    ----------
    model: nn.Module
    mode: must be "random_orthogonal"
    seed: random seed
    verbose: print summary

    Returns
    -------
    info = {"mode": str, "matrices": {in_features: Q}}
    """
    from models.quant.layers import QuantLinearW4A4

    if mode != "random_orthogonal":
        raise ValueError(f"Unknown rotation mode: {mode}")

    sizes = set()
    for _, m in model.named_modules():
        if isinstance(m, QuantLinearW4A4):
            sizes.add(m.in_features)

    dtype = next(m.weight.dtype for _, m in model.named_modules()
                  if isinstance(m, QuantLinearW4A4))
    device = next(m.weight.device for _, m in model.named_modules()
                   if isinstance(m, QuantLinearW4A4))

    cache = _build_random_orthogonal_cache(sizes, dtype, device)
    info: dict = {"mode": mode, "matrices": cache}

    count = 0
    for name, m in tqdm(list(model.named_modules()), desc="[rotate] random_orthogonal", disable=not verbose):
        if not isinstance(m, QuantLinearW4A4):
            continue
        Q = cache[m.in_features]
        _rotate_weight_dense(m.weight.data, Q)
        m.hadamard_signs = m.in_features
        m.register_forward_pre_hook(_make_dense_rotation_hook(Q))
        count += 1
    if verbose:
        print(f"[rotate] random_orthogonal: {count} layers (sizes: {sorted(sizes)})")

    return info
