import torch
import torch.nn as nn
import torch.nn.functional as F


class DownProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pool_factor: int):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_factor, stride=pool_factor)
        self.proj = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.proj.weight[:out_dim, :out_dim] = torch.eye(out_dim)
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        H = W = grid_hw
        x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.proj(x)
        return x


class BilinearUpsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        x = self.proj(self.norm(x))
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x


class DimProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            torch.nn.init.zeros_(self.proj.weight)
            self.proj.weight[:in_dim, :in_dim] = torch.eye(in_dim)
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class Bridge(nn.Module):
    def __init__(self, upsample: nn.Module = None, dim_proj: nn.Module = None):
        super().__init__()
        self.upsample = upsample
        self.dim_proj = dim_proj

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x, grid_hw)
        if self.dim_proj is not None:
            x = self.dim_proj(x)
        return x
