"""
W4A8 INT4 CUDA Extension for TinySR QuantLinearW4A4.

使用方法:
    from models.quant.int4_cuda import get_int4_linear_op
    int4_linear = get_int4_linear_op()
    output = int4_linear(x, packed_w, act_scale, wt_scale, bias)
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
        name="int4_linear_cuda",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_cuda_cflags=["-O3", "-arch=sm_80", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _int4_linear_module


def get_int4_linear_op():
    """返回编译好的 int4_linear CUDA 算子"""
    return _build_extension().forward


def int4_linear_quant_layer_forward(self, x):
    """
    替换 QuantLinearW4A4.forward 的 int4 推理路径。

    要求 layer 已通过 pack_all_quant_layers() 预处理，
    即 layer._int4_packed, layer._act_scale, layer._int4_wt_scale 已设置。
    """
    smooth_scale = getattr(self.weight_quantizer, "smooth_scale", None)
    if smooth_scale is not None:
        x = x / smooth_scale.reshape(*([1] * (x.dim() - 1)), -1)

    op = get_int4_linear_op()
    out = op(x, self._int4_packed, self._act_scale, self._int4_wt_scale,
             self.bias if self.bias is not None else torch.empty(0, device=x.device))

    if hasattr(self.weight_quantizer, "branch_forward"):
        branch_out = self.weight_quantizer.branch_forward(x)
        if branch_out is not None:
            out = out + branch_out

    return out
