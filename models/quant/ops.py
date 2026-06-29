import math

import torch
from typing import Union


@torch.no_grad()
def affine_fake_quant_weight(weight, bits=4, symmetric=True, eps=1e-8, group_size=-1):
    if group_size > 0 and weight.dim() >= 2:
        out_feat, in_feat = weight.shape
        if in_feat % group_size != 0:
            group_size = -1
        else:
            gs = group_size
            num_groups = in_feat // gs
            w_groups = weight.view(out_feat, num_groups, gs)
            if symmetric:
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
                scale = w_groups.abs().amax(dim=-1).div(float(qmax)).clamp_min(eps)
                zp = torch.zeros_like(scale)
            else:
                qmin = 0
                qmax = 2 ** bits - 1
                min_val = w_groups.amin(dim=-1)
                max_val = w_groups.amax(dim=-1)
                scale = (max_val - min_val).div(float(qmax - qmin)).clamp_min(eps)
                zp = (qmin - torch.round(min_val / scale)).clamp(qmin, qmax)
            w_int = torch.round(w_groups / scale.unsqueeze(-1) + zp.unsqueeze(-1))
            w_int = torch.clamp(w_int, qmin, qmax)
            return ((w_int - zp.unsqueeze(-1)) * scale.unsqueeze(-1)).view(out_feat, in_feat)

    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = weight.abs().amax(dim=1, keepdim=True).div(float(qmax)).clamp_min(eps)
        zero_point = torch.zeros_like(scale)
    else:
        qmin = 0
        qmax = 2 ** bits - 1
        min_val = weight.amin(dim=1, keepdim=True)
        max_val = weight.amax(dim=1, keepdim=True)
        scale = (max_val - min_val).div(float(qmax - qmin)).clamp_min(eps)
        zero_point = (qmin - torch.round(min_val / scale)).clamp(qmin, qmax)

    weight_int = torch.round(weight / scale + zero_point).clamp(qmin, qmax)
    return (weight_int - zero_point) * scale


@torch.no_grad()
def gptq_per_group_int4(weight, inputs, group_size=64, bits=4, symmetric=True,
                        block_size=128, damp_percentage=0.01, eps=1e-8):
    """Per-group GPTQ for nunchaku export. Returns (int4_weight, scales)."""
    out_feat, in_feat = weight.shape
    assert in_feat % group_size == 0
    num_groups = in_feat // group_size
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    work_weight = weight.float()
    work_inputs = inputs.float().to(device=weight.device)

    qweight_int = torch.zeros(out_feat, in_feat, dtype=torch.int32, device=weight.device)
    all_scales = torch.zeros(out_feat, num_groups, dtype=torch.float32, device=weight.device)

    for g in range(num_groups):
        g_start = g * group_size
        g_end = g_start + group_size
        w_g = work_weight[:, g_start:g_end].clone()
        x_g = work_inputs[:, g_start:g_end]

        scales_g = w_g.abs().amax(dim=1).div(float(qmax)).clamp_min(eps)

        H = x_g.T @ x_g
        dead = H.diagonal() == 0
        if dead.any():
            H[dead, dead] = 1
            w_g[:, dead] = 0

        importance = torch.diag(H)
        perm = torch.argsort(importance, descending=True)
        H = H[perm][:, perm]
        w_g = w_g[:, perm]

        H_diag = H.diagonal()
        H_diag += damp_percentage * H_diag.mean()
        H_inv = None
        for _ in range(200):
            try:
                L = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(L)
                H_inv = torch.linalg.cholesky(H_inv, upper=True)
                break
            except RuntimeError:
                H_diag += (damp_percentage * 0.1) * H_diag.mean()

        if H_inv is None:
            q_g = torch.round(w_g / scales_g.unsqueeze(1)).clamp(qmin, qmax)
        else:
            q_g = torch.zeros_like(w_g)
            for col in range(group_size):
                column = w_g[:, col]
                pos_diag = H_inv[col, col].clamp_min(eps)
                qcol = torch.round(column / scales_g).clamp(qmin, qmax)
                q_g[:, col] = qcol
                err = (column - qcol * scales_g) / pos_diag
                if col < group_size - 1:
                    w_g[:, col + 1:] -= err.unsqueeze(1) * H_inv[col, col + 1:].unsqueeze(0)

        q_g = q_g[:, torch.argsort(perm)]
        qweight_int[:, g_start:g_end] = q_g.to(torch.int32)
        all_scales[:, g] = scales_g

    return qweight_int, all_scales


@torch.no_grad()
def gptq_per_group_dequant(weight, inputs, group_size=64, bits=4, symmetric=True,
                           block_size=128, damp_percentage=0.01, eps=1e-8):
    """Per-group GPTQ for train_quant calibration, returns dequantized FP weights."""
    qw_int, scales = gptq_per_group_int4(
        weight, inputs, group_size=group_size, bits=bits, symmetric=symmetric,
        block_size=block_size, damp_percentage=damp_percentage, eps=eps)
    out_feat, in_feat = qw_int.shape
    num_groups = in_feat // group_size
    scales_expanded = scales.unsqueeze(-1).expand(out_feat, num_groups, group_size).reshape(out_feat, in_feat)
    return qw_int.float().mul_(scales_expanded).to(dtype=weight.dtype)


@torch.no_grad()
def gptq_quantize_linear_weight(
    weight,
    inputs,
    bits=4,
    symmetric=True,
    block_size=128,
    damp_percentage=0.01,
    eps=1e-8,
    group_size=-1,
):
    if inputs is None or inputs.numel() == 0:
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps, group_size=group_size)

    if group_size > 0:
        return gptq_per_group_dequant(weight, inputs, group_size=group_size, bits=bits,
                                      symmetric=symmetric, block_size=block_size,
                                      damp_percentage=damp_percentage, eps=eps)

    orig_dtype = weight.dtype
    work_weight = weight.float()
    work_inputs = inputs.to(device=weight.device, dtype=torch.float32)
    if work_inputs.ndim != 2 or work_inputs.shape[1] != work_weight.shape[1]:
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)

    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = work_weight.abs().amax(dim=1, keepdim=True).div(float(qmax)).clamp_min(eps)
        zero_point = torch.zeros_like(scale)
    else:
        qmin = 0
        qmax = 2 ** bits - 1
        min_val = work_weight.amin(dim=1, keepdim=True)
        max_val = work_weight.amax(dim=1, keepdim=True)
        scale = (max_val - min_val).div(float(qmax - qmin)).clamp_min(eps)
        zero_point = (qmin - torch.round(min_val / scale)).clamp(qmin, qmax)

    hessian = math.sqrt(2.0 / max(work_inputs.shape[0], 1)) * work_inputs
    hessian = hessian.t().matmul(hessian)
    dead = hessian.diagonal() == 0
    if dead.any():
        hessian[dead, dead] = 1
        work_weight[:, dead] = 0

    importance = torch.diag(hessian)
    permute = torch.argsort(importance, descending=True)
    inverse_permute = torch.argsort(permute)
    hessian = hessian[permute][:, permute]
    work_weight = work_weight[:, permute]

    hessian_diag = hessian.diagonal()
    hessian_diag_mean = hessian_diag.mean()
    hessian_diag += damp_percentage * hessian_diag_mean

    hessian_inv = None
    for _ in range(200):
        try:
            hessian_inv = torch.linalg.cholesky(hessian)
            hessian_inv = torch.cholesky_inverse(hessian_inv)
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
            break
        except RuntimeError:
            hessian_diag += (damp_percentage * 0.1) * hessian_diag_mean
    if hessian_inv is None:
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)

    qtensor = torch.zeros_like(work_weight)
    num_columns = work_weight.shape[1]
    for c_start in range(0, num_columns, block_size):
        c_end = min(c_start + block_size, num_columns)
        block_weight = work_weight[:, c_start:c_end].clone()
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]
        block_error = torch.zeros_like(block_weight)
        for local_col in range(c_end - c_start):
            column = block_weight[:, local_col]
            pos_diag = block_hessian_inv[local_col, local_col].clamp_min(eps)
            qcolumn = torch.round(column.view(-1, 1) /
                                   scale + zero_point).clamp(qmin, qmax)
            qcolumn = ((qcolumn - zero_point) * scale).view(-1)
            qtensor[:, c_start + local_col] = qcolumn
            column_error = (column - qcolumn) / pos_diag
            block_error[:, local_col] = column_error
            block_weight[:, local_col:] -= column_error.view(-1, 1).matmul(
                block_hessian_inv[local_col, local_col:].view(1, -1)
            )
        work_weight[:,
                    c_end:] -= block_error.matmul(hessian_inv[c_start:c_end, c_end:])

    qtensor = qtensor[:, inverse_permute]
    if qtensor.isnan().any() or qtensor.isinf().any():
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)
    return qtensor.to(orig_dtype)


@torch.no_grad()
def fake_quant_activation(x: torch.Tensor, bits: int = 8, symmetric: bool = True, eps: float = 1e-8,
                           scale: Union[torch.Tensor, float, None] = None,
                           group_size: int = -1) -> torch.Tensor:
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        zero_point = 0
    else:
        qmin = 0
        qmax = 2 ** bits - 1
        zero_point = None

    if group_size > 0 and x.dim() >= 2:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        N, K = x_flat.shape
        if K % group_size != 0:
            return x
        num_groups = K // group_size
        x_view = x_flat.view(N, num_groups, group_size)
        if symmetric:
            max_abs = x_view.abs().amax(dim=-1)
            s = max_abs / float(qmax)
            s = s.clamp_min(eps)
            zp = torch.zeros_like(s)
        else:
            min_val = x_view.amin(dim=-1)
            max_val = x_view.amax(dim=-1)
            s = (max_val - min_val) / float(qmax - qmin)
            s = s.clamp_min(eps)
            zp = qmin - torch.round(min_val / s)
            zp = zp.clamp(qmin, qmax)
        x_int = torch.round(x_view / s.unsqueeze(-1) + zp.unsqueeze(-1))
        x_int = torch.clamp(x_int, qmin, qmax)
        x_dequant = (x_int - zp.unsqueeze(-1)) * s.unsqueeze(-1)
        return x_dequant.view(*orig_shape)

    if scale is not None:
        scale = scale.item() if isinstance(scale, torch.Tensor) else scale
    else:
        max_abs = x.abs().max().clamp_min(eps)
        scale = max_abs / float(qmax)
    if not symmetric:
        scale = (x.max() - x.min()).clamp_min(eps) / float(qmax - qmin)
        zero_point = qmin - torch.round(x.min() / scale)
        zero_point = zero_point.clamp(qmin, qmax)
    x_int = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return (x_int - zero_point) * scale
