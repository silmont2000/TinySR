"""Rotation for outlier suppression.

Two modes:
- random_orthogonal: dense QR matrix, O(n^2) per forward pass, any n.
- fast_hadamard:    structured Walsh-Hadamard (fixed +1 signs), O(n log n) via FWHT.
                     Requires scipy. Uses local hadamard_math module.
"""

import math

import torch
from tqdm import tqdm


# -- Shared utilities ----------------------------------------------------

def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


_hadamard_utils_cache = None


def _get_hadamard_utils():
    """Lazy import of local Hadamard math utilities (cached)."""
    global _hadamard_utils_cache
    if _hadamard_utils_cache is not None:
        return _hadamard_utils_cache
    from models.quant.hadamard_math import get_hadamard_matrices, hardmard_transform
    _hadamard_utils_cache = (get_hadamard_matrices, hardmard_transform)
    return _hadamard_utils_cache


# -- Mode: random_orthogonal (dense QR, no padding) ---------------------

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


# -- Mode: fast_hadamard (structured Walsh-Hadamard) --------------------

def _build_hadamard_cache(sizes: set[int], dtype, device) -> dict:
    """Build Hadamard rotation cache.

    Returns
    -------
    {in_features: {"rhs": rhs, "lhs": lhs, "lhs_k": k}}
    """
    get_hadamard_matrices, _ = _get_hadamard_utils()
    cache = {}
    for n in sizes:
        rhs, lhs, k = get_hadamard_matrices(n, scale=True, dtype=torch.float32, device=device)
        cache[n] = {
            "rhs": rhs.to(dtype=dtype, device=device),
            "lhs": lhs.to(dtype=dtype, device=device),
            "lhs_k": k,
        }
    return cache


def _rotate_weight_hadamard(weight: torch.Tensor, rhs: torch.Tensor,
                            lhs: torch.Tensor, lhs_k: int) -> None:
    _, hardmard_transform = _get_hadamard_utils()
    transformed = hardmard_transform(
        weight.data.float(),
        hadamard_rhs=rhs.to(device=weight.device, dtype=torch.float32),
        hadamard_lhs=lhs.to(device=weight.device, dtype=torch.float32),
        lhs_k=lhs_k,
        scaled=True,
    )
    weight.copy_(transformed.to(dtype=weight.dtype, device=weight.device))


def _make_hadamard_rotation_hook(rhs: torch.Tensor, lhs: torch.Tensor, lhs_k: int):
    _, hardmard_transform = _get_hadamard_utils()

    def _hook(module, args):
        x = args[0]
        x = hardmard_transform(
            x.float(),
            hadamard_rhs=rhs.to(device=x.device, dtype=torch.float32),
            hadamard_lhs=lhs.to(device=x.device, dtype=torch.float32),
            lhs_k=lhs_k,
            scaled=True,
        )
        return (x.to(dtype=args[0].dtype), *args[1:])

    return _hook


# -- Unified entry point -------------------------------------------------

def enable_rotation(model, mode: str = "random_orthogonal", seed: int = 0,
                    verbose: bool = True) -> dict:
    """Apply orthogonal rotation to all QuantLinearW4A4 layers.

    Parameters
    ----------
    model: nn.Module
    mode: "random_orthogonal" (dense QR) or "fast_hadamard" (FWHT)
    seed: random seed (only used by random_orthogonal)
    verbose: print summary

    Returns
    -------
    info = {"mode": str, "matrices": dict}
      random_orthogonal: matrices = {in_features: Q_tensor}
      fast_hadamard:     matrices = {in_features: {"rhs": rhs, "lhs": lhs, "lhs_k": k}}
    """
    from models.quant.layers import QuantLinearW4A4

    if mode not in ("random_orthogonal", "fast_hadamard"):
        raise ValueError(f"Unknown rotation mode: {mode}")

    sizes: set[int] = set()
    for _, m in model.named_modules():
        if isinstance(m, QuantLinearW4A4):
            sizes.add(m.in_features)

    dtype = next(m.weight.dtype for _, m in model.named_modules()
                 if isinstance(m, QuantLinearW4A4))
    device = next(m.weight.device for _, m in model.named_modules()
                  if isinstance(m, QuantLinearW4A4))

    torch.manual_seed(seed)

    if mode == "random_orthogonal":
        cache = _build_random_orthogonal_cache(sizes, dtype, device)
    else:
        cache = _build_hadamard_cache(sizes, dtype, device)
    info: dict = {"mode": mode, "matrices": cache}

    desc = f"[rotate] {mode}"
    count = 0
    for name, m in tqdm(list(model.named_modules()), desc=desc, disable=not verbose):
        if not isinstance(m, QuantLinearW4A4):
            continue
        if mode == "random_orthogonal":
            Q = cache[m.in_features]
            _rotate_weight_dense(m.weight.data, Q)
            m.register_forward_pre_hook(_make_dense_rotation_hook(Q))
        else:
            entry = cache[m.in_features]
            _rotate_weight_hadamard(m.weight.data, entry["rhs"], entry["lhs"], entry["lhs_k"])
            m.register_forward_pre_hook(
                _make_hadamard_rotation_hook(entry["rhs"], entry["lhs"], entry["lhs_k"]))
        m.hadamard_signs = m.in_features
        count += 1
    if verbose:
        print(f"[rotate] {mode}: {count} layers (sizes: {sorted(sizes)})")

    return info
