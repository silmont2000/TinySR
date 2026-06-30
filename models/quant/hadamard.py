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


# ── Mode: fast_hadamard (Walsh-Hadamard + power-of-2 padding) ──────────

def _generate_fast_hadamard_signs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float16)


def _fast_hadamard_transform(x: torch.Tensor) -> None:
    """In-place Walsh-Hadamard transform on last dim. n must be power of 2."""
    n = x.shape[-1]
    h = 1
    while h < n:
        a = x[..., :h].clone()
        b = x[..., h : 2 * h]
        x[..., :h] = a + b
        x[..., h : 2 * h] = a - b
        h *= 2
    x.div_(math.sqrt(float(n)))


def _rotate_weight_hadamard(weight: torch.Tensor, signs: torch.Tensor) -> None:
    """W ← fast_hadamard(W * signs), in-place."""
    weight.mul_(signs.to(device=weight.device, dtype=weight.dtype))
    _fast_hadamard_transform(weight)


def _make_hadamard_rotation_hook(signs: torch.Tensor, original_dim: int):
    def _hook(module, args):
        x = args[0]
        if x.shape[-1] < signs.numel():
            pad = torch.zeros(*x.shape[:-1], signs.numel() - x.shape[-1],
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        x = x.clone() if x.requires_grad else x
        x.mul_(signs.to(device=x.device, dtype=x.dtype))
        _fast_hadamard_transform(x)
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
        for name, m in list(model.named_modules()):
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
        signs_map: dict[str, torch.Tensor] = {}
        padded_map: dict[str, int] = {}

        count = 0
        for name, m in list(model.named_modules()):
            if not isinstance(m, QuantLinearW4A4):
                continue
            n = m.in_features
            padded = _next_power_of_2(n)
            layer_seed = seed + hash(name) % 100000
            signs = _generate_fast_hadamard_signs(padded, seed=layer_seed)

            if padded != n:
                pad_size = padded - n
                pad = torch.zeros(m.out_features, pad_size, dtype=dtype, device=device)
                m.weight = torch.nn.Parameter(torch.cat([m.weight.data, pad], dim=1))

            _rotate_weight_hadamard(m.weight.data, signs.to(dtype=dtype, device=device))
            m.hadamard_signs = padded
            m.register_forward_pre_hook(_make_hadamard_rotation_hook(signs, original_dim=n))
            signs_map[name] = signs
            padded_map[name] = padded
            count += 1

        info["signs_map"] = signs_map
        info["padded_map"] = padded_map
        if verbose:
            print(f"[rotate] fast_hadamard: {count} layers "
                  f"(padded sizes: {sorted(set(padded_map.values()))})")

    else:
        raise ValueError(f"Unknown rotation mode: {mode}")

    return info
