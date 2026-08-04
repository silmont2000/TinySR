"""ViDiT-Q quantized linear layer: SmoothQuant + Hadamard rotation.

Adapted from ViDiT-Q's viditq_quant_layer.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quant.viditq_quant_layer import QuantizedLinear
from models.quant.viditq_quarot_utils import random_hadamard_matrix


class ViDiTQuantizedLinear(QuantizedLinear):
    """Quantized linear layer with channel-wise smooth scaling + Hadamard rotation.

    Forward path:
        1. x = x * channel_mask           (channel-wise smooth scaling)
        2. x = matmul(x, rotation_matrix)  (Hadamard rotation)
        3. x = a_quantizer(x)             (dynamic per-token activation quantization)
        4. y = F.linear(x, weight)         (dequantized weight forward)

    The weight is pre-processed during PTQ:
        W_q = quantize((W_fp / channel_mask) @ rotation_matrix)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device,
        quant_config: dict,
        fp_module: nn.Linear,
        alpha: float = 0.5,
    ):
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)
        self.alpha = alpha
        self.channel_mask = None
        self.rotation_matrix = None

    @torch.no_grad()
    def set_channel_mask(self, act_absmax):
        """Compute channel mask from activation statistics.

        channel_mask = (|W|^alpha) / (|act|^(1-alpha))

        Args:
            act_absmax: per-channel max absolute activation, shape [C_in].
        """
        weight_absmax = self.fp_module.weight.abs().max(dim=0)[0]
        channel_mask = (weight_absmax.abs() ** self.alpha) / (act_absmax.abs() ** (1 - self.alpha))
        self.channel_mask = channel_mask.to(dtype=torch.float64)

    @torch.no_grad()
    def set_rotation_matrix(self, device="cuda"):
        R = random_hadamard_matrix(self.in_features, device)
        self.rotation_matrix = R.to(torch.float16)
        del R

    @torch.no_grad()
    def update_quantized_weight(self):
        """Scale → quantize → rotate the weight (offline PTQ step)."""
        assert self.channel_mask is not None and self.rotation_matrix is not None
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False
        W_scaled = self.fp_module.weight / self.channel_mask.reshape([1, C_in])
        W_scaled_q = self.w_quantizer(W_scaled)
        rot_dtype = self.rotation_matrix.dtype
        W_rot = torch.matmul(W_scaled_q.to(rot_dtype), self.rotation_matrix).float()
        self.weight.data = self.w_quantizer(W_rot)
        self.w_quantizer.init_done = True
        self.fp_module = None  # free original FP weight after quantization

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if not self.quant_mode:
            return self.fp_module(x, *args, **kwargs)

        dtype_ = x.dtype
        B, N_token, C = x.shape
        x = x * self.channel_mask.reshape([1, 1, C])
        x = torch.matmul(x.to(self.rotation_matrix.dtype), self.rotation_matrix).to(dtype=dtype_)
        x = x.reshape([B * N_token, -1])
        x = self.a_quantizer(x)
        x = x.reshape([B, N_token, C])
        y = F.linear(x, self.weight.to(dtype=dtype_), self.bias, *args, **kwargs)
        return y
