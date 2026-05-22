"""
int4→fp16 dequant + smooth_scale CUDA extension.

Usage:
    from models.quant.int4_cuda import get_fused_prepare
    x_prep, w_deq = get_fused_prepare()(x, packed_w, w_scale, inv_smooth)
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_source_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_path = os.path.join(_source_dir, "csrc", "int4_linear_kernel.cu")
_cpp_path  = os.path.join(_source_dir, "csrc", "int4_linear.cpp")

_mod = None

def _build():
    global _mod
    if _mod is not None:
        return _mod
    with open(_cuda_path) as f: cu = f.read()
    with open(_cpp_path)  as f: cp = f.read()
    _mod = load_inline(
        name="int4_dequant_cuda",
        cpp_sources=[cp], cuda_sources=[cu],
        extra_cuda_cflags=["-O3", "-arch=sm_80", "--use_fast_math"],
        extra_cflags=["-O3"], verbose=False)
    return _mod

def get_fused_prepare():
    return _build().fused_prepare

def get_dequant_op():
    return _build().dequant
