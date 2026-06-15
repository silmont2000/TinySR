import torch
import torch.nn as nn
import torch.nn.functional as F


class DownProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pool_factor: int):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_factor, stride=pool_factor)
        # self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.proj.weight[:out_dim, :out_dim] = torch.eye(out_dim)
            self.proj.bias.zero_()
            # nn.init.zeros_(self.conv.weight)
            # nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        H = W = grid_hw
        x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)
        x = self.pool(x)
        # x = F.gelu(self.conv(x))
        # x = x + self.conv(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.proj(x)
        return x


class BilinearUpsample(nn.Module):
    def __init__(self, dim: int, scale_factor: float = 2.0):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.scale_factor = scale_factor
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        x = self.proj(x)
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x


class BilinearDownsample(nn.Module):
    def __init__(self, dim: int, scale_factor: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.scale_factor = scale_factor
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        x = self.proj(self.norm(x))
        x = F.gelu(x)
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
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
            if out_dim > in_dim:
                self.proj.weight[in_dim:, :] = torch.randn(out_dim - in_dim, in_dim) * 1e-3
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class Bridge(nn.Module):
    def __init__(self, upsample: nn.Module = None, downsample: nn.Module = None, dim_proj: nn.Module = None):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.dim_proj = dim_proj

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x, grid_hw)
        if self.downsample is not None:
            x = self.downsample(x, grid_hw)
        if self.dim_proj is not None:
            x = self.dim_proj(x)
        return x


class ConvBlock(nn.Module):
    """Depthwise-separable spatial convolution for token sequences.
    Inserts between p_states or before/after upsampling.
    
    (B, N, D) → reshape 2D → DepthwiseConv2d → GELU → Pointwise Linear."""
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.spatial = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=dim)
        self.mix = nn.Linear(dim, dim)
        with torch.no_grad():
            self.mix.weight.copy_(torch.eye(dim))
            self.mix.bias.zero_()

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        x = self.norm(x)
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)
        x = self.spatial(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.mix(x)
        return x


class ConvUpsample(nn.Module):
    """Convolution-based upsampling, replaces bilinear with ConvTranspose2d.
    Same interface as BilinearUpsample.

    LayerNorm → Linear(d,d) → reshape 2D → ConvTranspose2d(k, s=scale) → flatten."""
    def __init__(self, dim: int, scale_factor: int = 2, kernel_size: int = 4):
        super().__init__()
        stride = scale_factor
        padding = max(0, (kernel_size - stride) // 2)
        output_padding = max(0, stride - kernel_size)
        self.norm = nn.LayerNorm(dim)
        self.mix = nn.Linear(dim, dim)
        self.up = nn.ConvTranspose2d(dim, dim, kernel_size=kernel_size,
                                     stride=stride, padding=padding,
                                     output_padding=output_padding)
        with torch.no_grad():
            self.mix.weight.copy_(torch.eye(dim))
            self.mix.bias.zero_()

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        x = self.mix(self.norm(x))
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)
        x = self.up(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x

class BilinearResidualUpsample(nn.Module):
    def __init__(self, dim: int, scale_factor: float = 2.0):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.scale_factor = scale_factor
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        assert N == grid_hw * grid_hw, f"{N} != {grid_hw}^2"
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x_res = F.gelu(self.conv(x_up))
        x = x_up + x_res
        return x.permute(0, 2, 3, 1).flatten(1, 2)
        
class LatentUpsample(nn.Module):
    """Latent-space upsampling: 16-channel, supports bilinear or conv.
    
    (B, 16, H, W) → LayerNorm → upsample(2x) → (B, 16, 2H, 2W)
    """
    def __init__(self, channels: int = 16, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(channels)
        if mode == "conv":
            self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4,
                                            stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)              # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)              # (B, C, H, W)
        if self.mode == "conv":
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


# ---------------------------------------------------------------------------
# Pixel Shuffle bridge blocks  (lossless spatial rearrangement)
# ---------------------------------------------------------------------------

class PixelShuffleDownsample(nn.Module):
    """Downsample via pixel_unshuffle (space-to-depth) + learned channel compression.

    Input:  (B, N, D)  with N = (H*s)*(H*s)
    Output: (B, N/s², D)  after pixel_unshuffle → Conv1x1(D*s² → D)
    """
    def __init__(self, dim: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        s2 = scale_factor * scale_factor
        self.norm = nn.LayerNorm(dim)
        self.compress = nn.Conv2d(dim * s2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        x = self.norm(x)
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)  # (B, D, H, H)
        x = F.pixel_unshuffle(x, downscale_factor=self.scale_factor) # (B, D*s², H/s, H/s)
        x = self.compress(x)                                          # (B, D, H/s, H/s)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)                      # (B, N/s², D)
        return x

    def init_as_bilinear_ds(self):
        """Initialise compress to mimic bilinear downsample: average each 2x2 block."""
        s = self.scale_factor
        s2 = s * s
        D = self.compress.out_channels
        self.compress.weight.data.zero_()
        for f in range(D):
            for pos in range(s2):
                self.compress.weight.data[f, f + D * pos, 0, 0] = 1.0 / s2
        self.compress.bias.data.zero_()


class PixelShuffleUpsample(nn.Module):
    """Upsample via learned channel expansion + pixel_shuffle (depth-to-space).

    Input:  (B, N, D)
    Output: (B, N*s², D)  after Conv1x1(D → D*s²) → pixel_shuffle
    """
    def __init__(self, dim: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        s2 = scale_factor * scale_factor
        self.expand = nn.Conv2d(dim, dim * s2, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, grid_hw: int) -> torch.Tensor:
        B, N, D = x.shape
        x = self.norm(x)
        x = x.reshape(B, grid_hw, grid_hw, D).permute(0, 3, 1, 2)  # (B, D, H, H)
        x = self.expand(x)                                           # (B, D*s², H, H)
        x = F.pixel_shuffle(x, upscale_factor=self.scale_factor)     # (B, D, H*s, H*s)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)                     # (B, N*s², D)
        return x

    def init_as_nearest_us(self):
        """Initialise expand to mimic nearest-neighbour upsample: replicate each channel."""
        s = self.scale_factor
        s2 = s * s
        D = self.expand.in_channels
        self.expand.weight.data.zero_()
        for f in range(D):
            for pos in range(s2):
                self.expand.weight.data[f + D * pos, f, 0, 0] = 1.0
        self.expand.bias.data.zero_()
