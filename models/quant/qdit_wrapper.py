"""Q-DiT integration for TinySR: layer replacement + GPTQ calibration + PTQ.

Mirrors the ViDiT-Q integration pattern (models/quant/viditq_wrapper.py) and
the official Q-DiT `quantize_model_gptq` loop (scripts/quant_main.py):

  1. replace target nn.Linear layers with QDiTQuantLinear (weight quant +
     dynamic per-token input activation quant);
  2. per transformer block: collect GPTQ Hessians on the calibration data,
     then refine each layer's weight with `fasterquant` (block-wise error
     minimization = sequential per-block quantization, as in the official repo);
  3. the model is then ready for shared inference (`run_inference`).

Calibration data format is identical to the TinySR pipeline:
(list of (latent, timesteps, pooled_prompt_embeds, weight_dtype), forward_fn).
TDC (temporal discrepancy-aware calibration) reduces to single-timestep
calibration for the one-step SR model (see design doc).
"""

import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from models.quant.qdit_quant_layer import (
    QDiTQuantLinear,
    QLinearLayer,
    GPTQ,
    Quantizer_GPTQ,
    find_qlinear_layers,
)


class QDITArgs:
    """Lightweight args container matching Q-DiT's CLI defaults.

    Defaults follow the official `scripts/quant_main.sh`:
      --wbits 4 --abits 8 --act_group_size 128 --weight_group_size 128
      --use_gptq --quant_method max  (w_sym/a_sym off, clip_ratio 1.0)
    """

    def __init__(self, wbits=4, abits=8, w_sym=False, a_sym=False,
                 weight_group_size=128, act_group_size=128,
                 weight_channel_group=1, w_clip_ratio=1.0, a_clip_ratio=1.0,
                 quant_type="int", quant_method="max",
                 percdamp=0.01, gptq_blocksize=128, use_gptq=True):
        self.wbits = wbits
        self.abits = abits
        self.w_sym = w_sym
        self.a_sym = a_sym
        self.weight_group_size = weight_group_size
        self.act_group_size = act_group_size
        self.weight_channel_group = weight_channel_group
        self.w_clip_ratio = w_clip_ratio
        self.a_clip_ratio = a_clip_ratio
        self.quant_type = quant_type
        self.quant_method = quant_method
        self.percdamp = percdamp
        self.gptq_blocksize = gptq_blocksize
        self.use_gptq = use_gptq


def replace_with_qdit(
    module: nn.Module,
    target_suffixes,
    args: QDITArgs,
):
    """Replace nn.Linear layers (matching target_suffixes) with QDiTQuantLinear.

    Same suffix-matching logic as `replace_with_viditq` /
    `replace_linear_with_w4a4`. Skips layers with 'lora_' in the name.
    """
    replaced = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if 'lora_' in name:
            continue
        if target_suffixes is not None and not any(name.endswith(s) for s in target_suffixes):
            continue

        parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = module.get_submodule(parent_name) if parent_name else module

        q_linear = QDiTQuantLinear(
            originalLayer=child,
            args=copy.deepcopy(args),
        )
        setattr(parent, child_name, q_linear)
        replaced.append(name)

    print(f"[QDiT] replaced {len(replaced)} nn.Linear -> QDiTQuantLinear "
          f"(W{args.wbits}A{args.abits}, w_sym={args.w_sym}, a_sym={args.a_sym}, "
          f"wgs={args.weight_group_size}, ags={args.act_group_size})")
    return replaced


@torch.no_grad()
def quantize_model_qdit_gptq(module, calib_data_list, forward_fn, args: QDITArgs):
    """Per-block GPTQ refinement (faithful port of quantize_model_gptq).

    For each transformer block containing quantized layers: collect the
    Hessians of all its QLinearLayers over the calibration data, then run
    `fasterquant` per layer with the configured group size.
    """
    if not args.use_gptq:
        # plain per-channel / per-group weight quantization (Q-DiT quantize_model)
        for name, m in module.named_modules():
            if isinstance(m, QDiTQuantLinear):
                m.quant()
        print(f"[QDiT] plain weight quantization done (no GPTQ)")
        return

    transformer = _find_transformer(module)
    blocks = transformer.transformer_blocks
    print(f"[QDiT] GPTQ quantization over {len(blocks)} blocks ...")

    for i in tqdm(range(len(blocks)), desc="QDiT GPTQ per-block"):
        block = blocks[i]
        block_layers = find_qlinear_layers(block)
        if not block_layers:
            continue

        gptq = {}
        for name in block_layers:
            gptq[name] = GPTQ(block_layers[name])
            gptq[name].quantizer = Quantizer_GPTQ()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.w_sym, mse=False,
                channel_group=args.weight_channel_group,
                clip_ratio=args.w_clip_ratio,
                quant_type=args.quant_type,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in block_layers:
            handles.append(block_layers[name].register_forward_hook(add_batch(name)))

        # calibration forward (single timestep, fixed prompt) over LR latents
        with torch.no_grad():
            for model_input, ts, ppe, wd in calib_data_list:
                forward_fn(model_input, ts, ppe, wd)

        for h in handles:
            h.remove()

        for name in block_layers:
            gptq[name].fasterquant(
                percdamp=args.percdamp,
                blocksize=args.gptq_blocksize,
                groupsize=args.weight_group_size,
            )
            block_layers[name].quantized = True
            gptq[name].free()

        del gptq
        torch.cuda.empty_cache()

    print(f"[QDiT] GPTQ quantization finished")
    return module


def _find_transformer(module: nn.Module):
    """Locate the transformer root that holds `transformer_blocks`."""
    if hasattr(module, "transformer_blocks"):
        return module
    for m in module.modules():
        if hasattr(m, "transformer_blocks") and len(m.transformer_blocks) > 0:
            return m
    raise RuntimeError("[QDiT] could not locate transformer_blocks in module tree")
