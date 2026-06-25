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
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps, group_size=group_size)

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
