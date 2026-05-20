import torch


def pack_int4_per_channel(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 per-channel symmetric int4 量化后的 FP16 residual 打包成真正的 int4 格式。

    Args:
        residual: [out_features, in_features] FP16
                  值已经在 int4 量化网格上 (round(w/s) * s)

    Returns:
        packed: [out_features, in_features // 2] uint8
                每个 byte: low nibble = even col, high nibble = odd col
        scale:  [out_features] fp16 per-channel scale
    """
    assert residual.dim() == 2
    N, K = residual.shape
    assert K % 2 == 0, f"K={K} must be even for int4 packing"

    scale = residual.abs().amax(dim=1) / 7.0
    scale = scale.clamp_min(1e-10)

    int4_t = torch.round(residual / scale.view(-1, 1))
    int4_t = int4_t.clamp(-8, 7).to(torch.int8)

    iv = int4_t.view(N, K // 2, 2)
    packed = ((iv[:, :, 1].to(torch.int32) & 0xF) << 4) | \
             (iv[:, :, 0].to(torch.int32) & 0xF)
    packed = packed.to(torch.uint8)

    return packed, scale.half()


def pack_all_quant_layers(transformer):
    """
    遍历所有 QuantLinearW4A4 层，将 residual 打包为 int4 格式，
    注入到层属性中。

    Returns:
        n_packed: 成功打包的层数
    """
    n_packed = 0
    for name, m in transformer.named_modules():
        if not hasattr(m, 'weight_quantizer'):
            continue
        wq = m.weight_quantizer
        if not hasattr(wq, 'residual') or wq.residual is None:
            continue

        residual = wq.residual.detach()
        packed, scale = pack_int4_per_channel(residual)

        if hasattr(m, '_int4_packed') and not isinstance(m._int4_packed, torch.Tensor):
            delattr(m, '_int4_packed')
        if not hasattr(m, '_int4_packed'):
            m.register_buffer('_int4_packed', packed, persistent=False)
        else:
            m._int4_packed = packed

        if hasattr(m, '_int4_wt_scale') and not isinstance(m._int4_wt_scale, torch.Tensor):
            delattr(m, '_int4_wt_scale')
        if not hasattr(m, '_int4_wt_scale'):
            m.register_buffer('_int4_wt_scale', scale, persistent=False)
        else:
            m._int4_wt_scale = scale

        if hasattr(m, 'act_quantizer') and hasattr(m.act_quantizer, 'quantizer'):
            m._act_scale = m.act_quantizer.quantizer.scale.detach()
        else:
            m._act_scale = torch.tensor(1.0, device=residual.device, dtype=torch.float16)

        wq.residual = None
        if hasattr(m, 'weight') and m.weight is not None:
            del m.weight
        n_packed += 1

    return n_packed
