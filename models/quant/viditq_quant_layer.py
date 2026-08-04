"""Quantized linear layer — base class for ViDiT-Q.

Adapted from ViDiT-Q's quant_layer.py.  Plain dict config (no OmegaConf).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quant.viditq_base_quantizer import StaticQuantizer, DynamicQuantizer


class QuantizedLinear(nn.Linear):
    """Base quantized linear layer with static weight + dynamic activation quantization.

    Expects input shape [B, N_token, C] (3D).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device,
        quant_config: dict,
        fp_module: nn.Linear,
    ):
        super().__init__(in_features, out_features, bias, device)
        self.fp_module = fp_module
        self.q_cfg = quant_config
        self.w_quantizer = None
        self.a_quantizer = None

        if quant_config.get('weight', None) is not None:
            self.w_quantizer = StaticQuantizer(quant_config['weight'])
            self.weight.data = self.w_quantizer(fp_module.weight)
            self.w_quantizer.init_done = True
        else:
            self.weight.data = fp_module.weight

        self.fp_weight = self.fp_module.weight
        self.bias = fp_module.bias

        if quant_config.get('act', None) is not None:
            self.a_quantizer = DynamicQuantizer(quant_config['act'])

        self.quant_mode = True

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if not self.quant_mode:
            return self.fp_module(x, *args, **kwargs)
        B, N_token, C = x.shape
        x = x.reshape([B * N_token, -1])
        if self.a_quantizer is not None:
            x = self.a_quantizer(x)
        x = x.reshape([B, N_token, C])
        y = F.linear(x, self.weight, self.bias, *args, **kwargs)
        return y
