"""
int4→fp16 dequant CUDA extension for TinySR QuantLinearW4A4.

Usage:
    from models.quant.int4_cuda import get_dequant_op
    w_fp16 = get_dequant_op()(packed_weights, per_channel_scale)
"""

import os
import torch
from torch.utils.cpp_extension import load_inline


_source_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_source_path = os.path.join(_source_dir, "csrc", "int4_linear_kernel.cu")
_cpp_source_path = os.path.join(_source_dir, "csrc", "int4_linear.cpp")

_int4_linear_module = None


def _build_extension():
    global _int4_linear_module
    if _int4_linear_module is not None:
        return _int4_linear_module

    with open(_cuda_source_path, "r") as f:
        cuda_source = f.read()
    with open(_cpp_source_path, "r") as f:
        cpp_source = f.read()

    _int4_linear_module = load_inline(
        name="int4_dequant_cuda",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_cuda_cflags=["-O3", "-arch=sm_80", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _int4_linear_module


def get_dequant_op():
    """返回编译好的 int4→fp16 dequant 算子"""
    return _build_extension().dequant
