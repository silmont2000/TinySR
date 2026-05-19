import math

import torch
from typing import Union


@torch.no_grad()
def affine_fake_quant_weight(weight, bits=4, symmetric=True, eps=1e-8):
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
):
    if inputs is None or inputs.numel() == 0:
        return affine_fake_quant_weight(weight, bits=bits, symmetric=symmetric, eps=eps)

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
                           scale: Union[torch.Tensor, float, None] = None) -> torch.Tensor:
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        if scale is not None:
            scale = scale.item() if isinstance(scale, torch.Tensor) else scale
        else:
            max_abs = x.abs().max().clamp_min(eps)
            scale = max_abs / float(qmax)
        zero_point = 0
    else:
        qmin = 0
        qmax = 2 ** bits - 1
        scale = (x.max() - x.min()).clamp_min(eps) / float(qmax - qmin)
        zero_point = qmin - torch.round(x.min() / scale)
        zero_point = zero_point.clamp(qmin, qmax)
    x_int = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return (x_int - zero_point) * scale
