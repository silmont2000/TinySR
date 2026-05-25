from dataclasses import dataclass
from typing import Tuple

import functools

def print_property(func):
    @functools.wraps(func)
    def getter(self):
        result = func(self)
        print(f"{func.__name__}: {result}")
        return result
    return property(getter)

@dataclass
class PStateSpec:
    num_blocks: int
    dim: int
    grid_hw: int


@dataclass
class PyramidArchConfig:
    p_states: Tuple[PStateSpec, ...] = (
        PStateSpec(num_blocks=4, dim=768,  grid_hw=32),
        PStateSpec(num_blocks=4, dim=1152, grid_hw=32),
        PStateSpec(num_blocks=4, dim=1536, grid_hw=32),
    )
    patch_size: int = 2
    in_channels: int = 16
    sample_size: int = 64
    out_channels: int = 16
    pooled_projection_dim: int = 2048
    pos_embed_max_size: int = 96
    upsample_mode: str = "bilinear"  # "bilinear" | "conv"

    @property
    def num_total_blocks(self) -> int:
        return sum(ps.num_blocks for ps in self.p_states)

    @property
    def patch_embed_grid(self) -> int:
        return self.sample_size // self.patch_size

    @property
    def patch_embed_dim(self) -> int:
        return self.p_states[-1].dim

    @print_property 
    @property
    def need_down_proj(self) -> bool:
        return self.patch_embed_grid != self.p_states[0].grid_hw

    @print_property 
    @property
    def need_dim_proj(self) -> bool:
        return self.patch_embed_dim != self.p_states[0].dim

    def get_block_dim(self, global_idx: int) -> int:
        offset = 0
        for ps in self.p_states:
            if global_idx < offset + ps.num_blocks:
                return ps.dim
            offset += ps.num_blocks
        raise IndexError(f"global block index {global_idx} out of range")

    def to_dict(self) -> dict:
        return {
            "p_states": [
                {"num_blocks": ps.num_blocks, "dim": ps.dim, "grid_hw": ps.grid_hw}
                for ps in self.p_states
            ],
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "sample_size": self.sample_size,
            "out_channels": self.out_channels,
            "pooled_projection_dim": self.pooled_projection_dim,
            "pos_embed_max_size": self.pos_embed_max_size,
            "upsample_mode": self.upsample_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PyramidArchConfig":
        return cls(
            p_states=tuple(PStateSpec(**ps) for ps in d["p_states"]),
            patch_size=d.get("patch_size", 2),
            in_channels=d.get("in_channels", 16),
            sample_size=d.get("sample_size", 64),
            out_channels=d.get("out_channels", 16),
            pooled_projection_dim=d.get("pooled_projection_dim", 2048),
            pos_embed_max_size=d.get("pos_embed_max_size", 96),
            upsample_mode=d.get("upsample_mode", "bilinear"),
        )
