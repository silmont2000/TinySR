import sys
sys.path.append(".")

from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
# from diffusers.models.embeddings import PatchEmbed

from models.tinysr.tinysd3 import PatchEmbed
from models.tinysr.tinysd3 import AdaLayerNormContinuous
from models.tinysr.tinysd3block import JointTransformerBlock
from models.tinysr.pyramid_config import PyramidArchConfig, PStateSpec
from models.tinysr.pyramid_blocks import DownProj, BilinearUpsample, BilinearDownsample, DimProj, Bridge, ConvUpsample, BilinearResidualUpsample

logger = logging.get_logger(__name__)


class PStateGroup(nn.Module):
    def __init__(self, blocks: nn.ModuleList, bridge: Optional[Bridge] = None):
        super().__init__()
        self.blocks = blocks
        self.bridge = bridge


class TinyPyramidSD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        pyramid_config: Dict[str, Any] = None,
        patch_size: int = 2,
        in_channels: int = 16,
        sample_size: int = 128,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
    ):
        super().__init__()

        self.pyramid_config = PyramidArchConfig.from_dict(pyramid_config)
        pc = self.pyramid_config
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.final_dim = pc.p_states[-1].dim

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=pc.patch_size,
            in_channels=pc.in_channels,
            embed_dim=self.final_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.final_dim, pooled_projection_dim=pooled_projection_dim
        )

        if pc.need_down_proj or pc.need_dim_proj:
            pool_factor = pc.patch_embed_grid // pc.p_states[0].grid_hw if pc.need_down_proj else 1
            self.down_proj = DownProj(
                in_dim=self.final_dim,
                out_dim=pc.p_states[0].dim,
                pool_factor=pool_factor,
            )
        else:
            self.down_proj = None

        self.p_states = nn.ModuleList()
        for i, spec in enumerate(pc.p_states):
            blocks = nn.ModuleList([
                JointTransformerBlock(
                    dim=spec.dim,
                    num_attention_heads=spec.dim // 64,
                    attention_head_dim=spec.dim,
                    context_pre_only=False,
                )
                for _ in range(spec.num_blocks)
            ])

            if i == len(pc.p_states) - 1:
                bridge = None
            else:
                next_spec = pc.p_states[i + 1]
                upsample = None
                downsample = None
                dim_proj = None

                if spec.grid_hw < next_spec.grid_hw:
                    scale = next_spec.grid_hw / spec.grid_hw
                    if pc.upsample_mode == "conv":
                        upsample = BilinearResidualUpsample(dim=spec.dim, scale_factor=scale)
                    else:
                        upsample = BilinearUpsample(dim=spec.dim, scale_factor=scale)
                elif spec.grid_hw > next_spec.grid_hw:
                    scale = spec.grid_hw / next_spec.grid_hw
                    downsample = BilinearDownsample(dim=spec.dim, scale_factor=1.0 / scale)

                if spec.dim != next_spec.dim:
                    dim_proj = DimProj(spec.dim, next_spec.dim)
                bridge = Bridge(upsample=upsample, downsample=downsample, dim_proj=dim_proj)

            self.p_states.append(PStateGroup(blocks=blocks, bridge=bridge))

        self.norm_out = AdaLayerNormContinuous(self.final_dim, self.final_dim, elementwise_affine=False, eps=1e-6)
        self.norm_out_norm = self.norm_out.norm
        self.norm = nn.LayerNorm(self.final_dim, eps=1e-6)
        self.register_buffer("scale", torch.tensor([]))
        self.register_buffer("shift", torch.tensor([]))

        self.proj_out = nn.Linear(self.final_dim, pc.patch_size * pc.patch_size * self.out_channels, bias=True)
        self.initialized = False

    def _unpatchify(self, h, grid_hw):
        patch_size = self.pyramid_config.patch_size
        B = h.shape[0]
        h = h.reshape(B, grid_hw, grid_hw, patch_size, patch_size, self.out_channels)
        h = torch.einsum("nhwpqc->nchpwq", h)
        return h.reshape(B, self.out_channels, grid_hw * patch_size, grid_hw * patch_size)

    def _patchify(self, latent, grid_hw):
        B, C, H, W = latent.shape
        h = self.pos_embed.proj(latent)
        h = h.flatten(2).transpose(1, 2)
        pe = self.pos_embed.pos_embed[:, :h.shape[1], :]
        return h + pe

    def _tokens_to_latent(self, h: torch.Tensor, grid_hw: int) -> torch.Tensor:
        h = self.norm(h)
        h = self.proj_out(h)
        return self._unpatchify(h, grid_hw)

    def enable_forward_chunking(self, chunk_size=None, dim=0):
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
        chunk_size = chunk_size or 1

        def fn_recursive(module, chunk_size, dim):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
            for child in module.children():
                fn_recursive(child, chunk_size, dim)

        for module in self.children():
            fn_recursive(module, chunk_size, dim)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        processors = {}

        def fn_recursive_add_processors(name, module, processors):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name, module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        self.original_attn_processors = None
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")
        self.original_attn_processors = self.attn_processors
        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if self.training:
            self.temb = self.time_text_embed(timestep, pooled_projections)
        elif self.initialized is False:
            temb = self.time_text_embed(timestep, pooled_projections)
            del self.time_text_embed
            self.temb = temb
            print("del self.time_text_embed")

        pc = self.pyramid_config
        # h是贯穿始终的
        h = self._patchify(hidden_states, pc.p_states[0].grid_hw)

        if self.down_proj is not None:
            h = self.down_proj(h, pc.patch_embed_grid)

        pre_last_tokens = None
        h_after_bridge = None
        self._distill_q = None
        self._distill_k = None
        self._distill_v = None
        self._distill_h = None

        for i, p_state in enumerate(self.p_states):
            spec = pc.p_states[i]

            if i == len(self.p_states) - 1:
                pre_last_tokens = h

            for j, block in enumerate(p_state.blocks):
                # return_qkv = (self.training and i == 1 and j == 0)
                return_qkv = (self.training and i == len(self.p_states)-1 and j == len(p_state.blocks)-1)
                if self.training or self.initialized is False:
                    temb_i = self.temb[:, :spec.dim]
                    if return_qkv:
                        h, q, k, v = block(
                            hidden_states=h, temb=temb_i, return_qkv=True)
                        self._distill_q = q
                        self._distill_k = k
                        self._distill_v = v
                        self._distill_h = h
                    else:
                        h = block(hidden_states=h, temb=temb_i)
                else:
                    if return_qkv:
                        h, q, k, v = block.forward_(
                            hidden_states=h, return_qkv=True)
                        self._distill_q = q
                        self._distill_k = k
                        self._distill_v = v
                        self._distill_h = h
                    else:
                        h = block.forward_(hidden_states=h)

            if i < len(self.p_states) - 1:
                bridge = p_state.bridge
                if bridge is not None:
                    h = bridge(h, spec.grid_hw)
                h_after_bridge = h

        if self.training:
            h, scale, shift = self.norm_out(h, self.temb[:, :self.final_dim])
        elif self.initialized is False:
            h, scale, shift = self.norm_out(h, self.temb[:, :self.final_dim])
            self.scale = scale.detach()
            self.shift = shift.detach()
            self.initialized = True
            del self.norm_out
            del self.temb
            print("del self.norm_out")
            print("del self.temb")
        else:
            h = self.norm_out_norm(h) * (1 + self.scale)[:, None, :] + self.shift[:, None, :]
        h = self.proj_out(h)
        output = self._unpatchify(h, pc.p_states[-1].grid_hw)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, pre_last_tokens, h_after_bridge)
        return Transformer2DModelOutput(sample=output), pre_last_tokens, h_after_bridge

    # ------------------------------------------------------------------
    #   Weight loading from flat (pruned) 12-block checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_flat_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        pyramid_config: PyramidArchConfig,
        subfolder: str = "transformer",
        low_cpu_mem_usage: bool = False,
        ignore_mismatched_sizes: bool = True,
        torch_dtype=None,
        cache_dir=None,
        **kwargs,
    ):
        from models.tinysr.tinysd3 import TinySD3Transformer2DModel

        flat = TinySD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            low_cpu_mem_usage=low_cpu_mem_usage,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **kwargs,
        )

        pc = pyramid_config

        instance = cls(
            pyramid_config=pc.to_dict(),
            patch_size=pc.patch_size,
            in_channels=pc.in_channels,
            sample_size=pc.sample_size,
            pooled_projection_dim=pc.pooled_projection_dim,
            out_channels=pc.out_channels,
            pos_embed_max_size=pc.pos_embed_max_size,
        )

        instance._copy_truncated_weights(flat, pc)

        instance.pos_embed.proj.load_state_dict(flat.pos_embed.proj.state_dict())
        if flat.pos_embed.norm is not None and instance.pos_embed.norm is not None:
            instance.pos_embed.norm.load_state_dict(flat.pos_embed.norm.state_dict())
        instance.time_text_embed.load_state_dict(flat.time_text_embed.state_dict())
        instance.norm_out.load_state_dict(flat.norm_out.state_dict())
        instance.proj_out.load_state_dict(flat.proj_out.state_dict())


        return instance

    def _copy_truncated_weights(self, flat_model, pc: PyramidArchConfig):
        mapping = self._build_block_mapping(pc)
        src_dim = flat_model.transformer_blocks[0].attn.to_q.weight.shape[0]

        for flat_idx, pyr_info in mapping:
            pyr_state_idx, pyr_block_idx, tgt_dim = pyr_info
            flat_block = flat_model.transformer_blocks[flat_idx]
            pyr_block = self.get_block(pyr_state_idx, pyr_block_idx)

            flat_sd = flat_block.state_dict()
            pyr_sd = pyr_block.state_dict()

            truncated_sd = {}
            for key, tensor in flat_sd.items():
                if key not in pyr_sd:
                    continue
                target_tensor = pyr_sd[key]
                if tensor.shape == target_tensor.shape:
                    truncated_sd[key] = tensor
                else:
                    truncated = truncate_tensor(tensor, src_dim, tgt_dim)
                    if truncated.shape == target_tensor.shape:
                        truncated_sd[key] = truncated

            pyr_block.load_state_dict(truncated_sd, strict=False)

    @staticmethod
    def _build_block_mapping(pc: PyramidArchConfig) -> List[Tuple[int, Tuple[int, int, int]]]:
        mapping = []
        flat_idx = 0
        for state_idx, spec in enumerate(pc.p_states):
            for block_idx in range(spec.num_blocks):
                if flat_idx >= 12:
                    break
                mapping.append((flat_idx, (state_idx, block_idx, spec.dim)))
                flat_idx += 1
        return mapping

    def get_block(self, state_idx: int, block_idx: int):
        return self.p_states[state_idx].blocks[block_idx]

    def merge_and_unload(self, progressbar=False, safe_merge=False, adapter_names=None):
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar=False,
        safe_merge=False,
        adapter_names=None,
    ):
        from tqdm import tqdm
        from peft.tuners.tuners_utils import onload_layer
        from peft.utils import _get_submodules, ModulesToSaveWrapper

        key_list = [key for key, _ in self.named_modules() if "lora_" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)
        return self

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias
        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)
        for name, module in new_module.named_modules():
            if ("lora_" in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)


def truncate_tensor(tensor: torch.Tensor, src_dim: int, tgt_dim: int) -> torch.Tensor:
    slices = []
    for size in tensor.shape:
        if size > 0 and size % src_dim == 0:
            mult = size // src_dim
            slices.append(slice(0, mult * tgt_dim))
        else:
            slices.append(slice(None))
    return tensor[tuple(slices)]
