from peft import LoraConfig
from models.tinysr.pyramid_config import PyramidArchConfig, PStateSpec

# Paths
CKPT = "checkpoint/tinybackbone/prune-12-merge-tinysr"
VAE_CKPT = "checkpoint/vae/separable"
POOL_EMBED_PATH = "dataset/default/pool_embeds.pt"
DEFAULT_TIMESTEP = 1000

        # PStateSpec(num_blocks=4, dim=768,  grid_hw=8),
        # PStateSpec(num_blocks=4, dim=1152, grid_hw=16),
        # PStateSpec(num_blocks=4, dim=1536, grid_hw=32),
# Pyramid architecture (shared by training & validation)
DEFAULT_PYRAMID_CONFIG = PyramidArchConfig(
    p_states=(
        PStateSpec(12, 1536, 32),
    ),
    sample_size=64,
    upsample_mode="conv",
)
# DEFAULT_PYRAMID_CONFIG = PyramidArchConfig(
#     p_states=(
#         PStateSpec(2, 1536, 32),
#         PStateSpec(6, 1536, 16),
#         PStateSpec(4, 1536, 32),
#     ),
#     sample_size=64,
#     upsample_mode="conv",
# )
# DEFAULT_PYRAMID_CONFIG = PyramidArchConfig(
#     p_states=(
#         PStateSpec(7, 1536, 16),   # 5 blocks, 计算在 256px 等效分辨率
#         PStateSpec(5, 1536, 32),   # 0 block, 仅用于 unpatchify 出 512px
#     ),
#     sample_size=32,      #64*8=512， 32*8=256
#     upsample_mode="bilinear",
# )

# Per-block FFN expansion multiplier (key = global block index 0..11)
# Derived from SVD effective rank + per-channel activation concentration.
#   blk 0: SVD=0.562, Act=0.782 → mult=2.5  (most redundant)
#   blk 1-9: typical values → mult=3.0
#   S1(16×16) blocks 2-7: downsampled stage, can be aggressive → mult=2.5
#   blk 10: SVD≈0.70, Act=0.650 (highest sparsity) → mult=2.5
#   blk 11: SVD≈0.74, dense   → mult=3.0
PYRAMID_MULT_CONFIG = {
    0: 2.5, 1: 3.0,
    2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5, 6: 2.5, 7: 2.5,
    8: 3.0, 9: 3.0, 10: 2.5, 11: 3.0,
}
# PYRAMID_MULT_CONFIG = {}   # 取消注释以禁用（全部使用默认 mult=4）

# LoRA
LORA_R = 64
LORA_TARGET_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0",
    "proj", "linear", "linear_1", "linear_2", "net.2",
]

SMOKE_RANK_PATTERN = {"p_states.0": 64, "p_states.1": 64, "p_states.2": 64}
VALIDATE_RANK_PATTERN = SMOKE_RANK_PATTERN


def make_lora_config(rank_pattern=None):
    return LoraConfig(
        r=LORA_R, lora_alpha=LORA_R, init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
        rank_pattern=rank_pattern or dict(SMOKE_RANK_PATTERN),
    )


def input_size(pc: PyramidArchConfig) -> int:
    return pc.sample_size * 8
