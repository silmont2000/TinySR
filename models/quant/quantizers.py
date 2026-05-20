import torch
import torch.nn as nn


class MinMaxObserver(nn.Module):
    def __init__(self, per_channel=False, ch_axis=0):
        super().__init__()
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.enabled = True

    @torch.no_grad()
    def forward(self, x):
        if not self.enabled:
            return x

        if self.per_channel:
            ch_axis = self.ch_axis % x.dim()
            reduce_dims = [i for i in range(x.dim()) if i != ch_axis]
            cur_min = x.amin(dim=reduce_dims)
            cur_max = x.amax(dim=reduce_dims)

            if self.min_val.numel() == 1:
                self.min_val = cur_min.detach()
                self.max_val = cur_max.detach()
            else:
                self.min_val = torch.minimum(self.min_val, cur_min.detach())
                self.max_val = torch.maximum(self.max_val, cur_max.detach())
        else:
            self.min_val = torch.minimum(self.min_val, x.min().detach())
            self.max_val = torch.maximum(self.max_val, x.max().detach())

        return x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        bits=4,
        symmetric=True,
        per_channel=False,
        ch_axis=0,
        eps=1e-8,
    ):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.eps = eps
        self.enabled = True
        self.observer_enabled = True

        self.observer = MinMaxObserver(
            per_channel=per_channel, ch_axis=ch_axis)

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.calibrated = False

    @property
    def qmin(self):
        if self.symmetric:
            return -(2 ** (self.bits - 1))
        return 0

    @property
    def qmax(self):
        if self.symmetric:
            return 2 ** (self.bits - 1) - 1
        return 2 ** self.bits - 1

    @torch.no_grad()
    def calculate_qparams(self):
        min_val = self.observer.min_val
        max_val = self.observer.max_val

        if self.symmetric:
            max_abs = torch.maximum(min_val.abs(), max_val.abs())
            scale = max_abs / float(self.qmax)
            scale = scale.clamp_min(self.eps)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / float(self.qmax - self.qmin)
            scale = scale.clamp_min(self.eps)
            zero_point = self.qmin - torch.round(min_val / scale)
            zero_point = zero_point.clamp(self.qmin, self.qmax)

        self.scale = scale.detach()
        self.zero_point = zero_point.detach()
        self.calibrated = True

    def reshape_qparams(self, x):
        scale = self.scale
        zero_point = self.zero_point

        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.ch_axis % x.dim()] = -1
            scale = scale.reshape(shape)
            zero_point = zero_point.reshape(shape)

        return scale, zero_point

    def forward(self, x):
        if self.observer_enabled:
            self.observer(x)

        if not self.enabled:
            return x

        if not self.calibrated:
            self.calculate_qparams()

        if torch.isinf(self.scale).any() or torch.isnan(self.scale).any():
            return x

        scale, zero_point = self.reshape_qparams(x)
        # return x

        x_int = torch.round(x / scale + zero_point)
        x_int = torch.clamp(x_int, self.qmin, self.qmax)
        x_dequant = (x_int - zero_point) * scale
        return x_dequant
