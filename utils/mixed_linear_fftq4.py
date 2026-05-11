import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import bitsandbytes as bnb

class MixedLinearFFTQ4(nn.Module):
    """
    接入 bitsandbytes 算子的 FFT-MixedQ4 混合精度线性层。
    1. 分支 A (Low Freq): FP16/FP32 稠密计算。
    2. 分支 B (High Freq): 使用 bitsandbytes 4bit 算子进行加速。
    """
    def __init__(self, in_features, out_features, bias=True, high_freq_bits=4, compute_dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_bits = high_freq_bits
        
        # 分支 A: 低频分支 (FP)
        # self.register_parameter("w_low", nn.Parameter(torch.empty(out_features, in_features)))
        self.register_parameter("w_low", nn.Parameter(torch.empty(out_features, in_features, dtype=compute_dtype)))
        # 分支 B: 高频分支 (INT4)
        # 使用 bitsandbytes 的 4bit 线性层
        self.high_linear = bnb.nn.Linear4bit(
            in_features, 
            out_features, 
            bias=False, 
            compute_dtype=compute_dtype,
            quant_type="fp4" if high_freq_bits == 4 else "nf4" # 默认为 4bit
        )
        
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.empty(out_features)))
        else:
            self.register_parameter("bias", None)
            
        self.meta = {}

    def forward(self, x):
        print("DEBUG dtypes:", x.dtype, self.w_low.dtype, self.high_linear.weight.dtype if hasattr(self.high_linear, "weight") else "n/a")
        # 1. 低频分支计算
        # y_low = F.linear(x, self.w_low)
        y_low = F.linear(x, self.w_low.to(x.dtype))
        
        # 2. 高频分支计算 (真实使用 bitsandbytes 4bit 算子加速)
        y_high = self.high_linear(x)
        
        out = y_low + y_high
        if self.bias is not None:
            out += self.bias
        return out

    @staticmethod
    def from_linear(linear_module, budget_ratio, high_freq_bits=4):
        """
        从普通 Linear 层转换。
        """
        W = linear_module.weight.detach()
        dev = W.device
        dtype = W.dtype
        out_f, in_f = W.shape
        
        # 1. 频域分解
        F_coeffs = torch.fft.fft2(W.float())
        mag = F_coeffs.abs()
        
        total = W.numel()
        num_low = max(1, int(budget_ratio * total))
        if num_low < total:
            threshold = torch.topk(mag.flatten(), num_low).values[-1]
            mask = mag >= threshold
        else:
            mask = torch.ones_like(mag, dtype=torch.bool)
            
        W_low_f = F_coeffs * mask
        W_high_f = F_coeffs * (~mask)
        
        w_low_val = torch.fft.ifft2(W_low_f).real.to(dtype)
        w_high_val = torch.fft.ifft2(W_high_f).real.to(dtype)
        
        # 2. 初始化模块 (先不移动到设备，保持 float 状态进行拷贝)
        mixed = MixedLinearFFTQ4(in_f, out_f, linear_module.bias is not None, high_freq_bits, compute_dtype=dtype)
        
        # 3. 填充权重
        mixed.w_low.data.copy_(w_low_val)
        
        # 将高频部分注入 bnb.nn.Linear4bit
        # 注意：必须在 .to(dev) 之前拷贝，因为 bnb 在移动到 GPU 时会进行量化打包，
        # 之后 weight.data 的形状会变成 1D 的 packed 字节数组。
        with torch.no_grad():
            mixed.high_linear.weight.data.copy_(w_high_val)
            
        if linear_module.bias is not None:
            mixed.bias.data.copy_(linear_module.bias.detach())
            
        # 4. 移动到设备 (此时触发 bnb 的自动量化打包)
        mixed.to(device=dev, dtype=dtype)
            
        # 4. 统计成本
        fp_size = 2 if dtype == torch.float16 else 4
        orig_bytes = total * fp_size
        
        # bnb 4bit 存储成本：每个参数 0.5 bytes (4 bits) + 额外的量化元数据 (约 1/32~1/64)
        # 这里简化计算为理论值
        low_bytes = total * fp_size
        high_bytes = total * 0.5 
        bias_bytes = mixed.bias.numel() * fp_size if mixed.bias is not None else 0
        
        curr_bytes = low_bytes + high_bytes + bias_bytes
        
        mixed.meta = {
            "spectral_budget_ratio": budget_ratio,
            "high_freq_bits": 4, # bnb 目前主要是 4bit
            "orig_params": total,
            "impl_byte_ratio": curr_bytes / orig_bytes,
            "backend": "bitsandbytes_4bit"
        }
        
        return mixed
