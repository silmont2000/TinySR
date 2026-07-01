"""Random orthogonal rotation for outlier suppression.

Two modes:
  - "random_orthogonal" (default): dense QR matrix, no padding, O(n²) per forward pass.
    Works for any in_features size.  1536 stays 1536, 6144 stays 6144.
  - "fast_hadamard": randomized Walsh-Hadamard with power-of-2 padding.
    O(n log n) per forward pass, ~0.3% compute overhead at inference.
    Weights are padded to nearest 2^k (1536→2048, 6144→8192), rotated in place.
    Hook pads input, applies fast Hadamard.  Orthogonality guarantees correctness.

Both modes are mathematically equivalent: apply an orthogonal rotation
to both weights (calibration) and inputs (inference hook), cancelling out
to preserve the original linear layer output.
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


# ── Mode: fast_hadamard (factorized Walsh-Hadamard, no padding) ─────────

def _fast_hadamard_transform(x: torch.Tensor) -> None:
    """In-place Walsh-Hadamard on the last dim. n must be power of 2."""
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = x[..., i : i + h].clone()
            b = x[..., i + h : i + 2 * h]
            x[..., i : i + h] = a + b
            x[..., i + h : i + 2 * h] = a - b
        h *= 2
    x.div_(math.sqrt(float(n)))


def _largest_pow2_divisor(n: int) -> int:
    """Return the largest power-of-2 divisor of n."""
    p = 1
    while n % 2 == 0:
        n //= 2
        p *= 2
    return p


def _build_factorized_cache(sizes: set[int], dtype, device) -> dict[int, torch.Tensor]:
    """Build {k: L_matrix} cache where L is a (k,k) random orthogonal matrix.
    Each dimension n = k * n_div_k where n_div_k is the power-of-2 part.
    """
    cache = {}
    for n in sizes:
        n_div_k = _largest_pow2_divisor(n)
        k = n // n_div_k
        if k > 1 and k not in cache:
            R = torch.randn(k, k, dtype=torch.float32)
            Q, _ = torch.linalg.qr(R)
            cache[k] = Q.to(dtype=dtype).to(device=device)
    return cache


def _factorized_transform(x: torch.Tensor, L: torch.Tensor | None) -> None:
    """Apply (L ⊗ H) on the last two dimensions of x in-place.
    x shape: (..., k, n_div_k) where n_div_k must be power of 2.
    """
    _fast_hadamard_transform(x)         # apply H on dim=-1
    if L is not None:                   # apply L on dim=-2
        x.copy_(torch.einsum("ji,...is->...js", L.to(dtype=x.dtype, device=x.device), x))


def _rotate_weight_hadamard_factorized(weight: torch.Tensor, L: torch.Tensor | None,
                                       n_div_k: int) -> None:
    """W ← (L ⊗ H) applied to input channels, in-place. No padding."""
    k = weight.shape[1] // n_div_k
    w = weight.view(-1, k, n_div_k)
    _factorized_transform(w, L)
    weight.copy_(w.reshape(weight.shape[0], -1))


def _make_hadamard_factorized_hook(L: torch.Tensor | None, n_div_k: int):
    def _hook(module, args):
        x = args[0]
        k = x.shape[-1] // n_div_k
        x = x.reshape(-1, k, n_div_k)
        x = x.clone() if x.requires_grad else x
        _factorized_transform(x, L)
        x = x.reshape(args[0].shape)
        return (x, *args[1:])
    return _hook


# ── Unified entry point ─────────────────────────────────────────────────

def enable_rotation(model, mode: str = "random_orthogonal", seed: int = 0,
                    verbose: bool = True) -> dict:
    """Apply rotation to all QuantLinearW4A4 layers.

    Parameters
    ----------
    model: nn.Module
    mode: "random_orthogonal" | "fast_hadamard"
    seed: random seed
    verbose: print summary

    Returns
    -------
    info = {
        "mode": str,
        "matrices": {in_features: Q} for random_orthogonal,
        OR
        "signs_map": {layer_name: signs}, "padded_map": {layer_name: padded_size} for fast_hadamard
    }
    """
    from models.quant.layers import QuantLinearW4A4

    sizes = set()
    for _, m in model.named_modules():
        if isinstance(m, QuantLinearW4A4):
            sizes.add(m.in_features)

    dtype = next(m.weight.dtype for _, m in model.named_modules()
                 if isinstance(m, QuantLinearW4A4))
    device = next(m.weight.device for _, m in model.named_modules()
                  if isinstance(m, QuantLinearW4A4))

    info: dict = {"mode": mode}

    if mode == "random_orthogonal":
        cache = _build_random_orthogonal_cache(sizes, dtype, device)
        info["matrices"] = cache

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

    elif mode == "fast_hadamard":
        cache = _build_factorized_cache(sizes, dtype, device)

        count = 0
        for name, m in tqdm(list(model.named_modules()), desc="[rotate] fast_hadamard", disable=not verbose):
            if not isinstance(m, QuantLinearW4A4):
                continue
            n = m.in_features
            n_div_k = _largest_pow2_divisor(n)
            k = n // n_div_k
            L = cache.get(k) if k > 1 else None

            _rotate_weight_hadamard_factorized(m.weight.data, L, n_div_k)
            m.hadamard_signs = n
            m.register_forward_pre_hook(_make_hadamard_factorized_hook(L, n_div_k))
            count += 1

        info["factorized"] = True
        info["factorized_cache"] = cache
        if verbose:
            print(f"[rotate] fast_hadamard (factorized): {count} layers "
                  f"(sizes: {sorted(sizes)})")

    else:
        raise ValueError(f"Unknown rotation mode: {mode}")

    return info
