"""PTQ4DiT integration for TinySR: layer replacement + calibration + block recon.

Faithful port of the official PTQ4DiT pipeline (quant_sample.py) onto the
TinySR SD3-based backbone:

  1. replace target nn.Linear layers with QuantModule (asymmetric per-channel
     weight quant with 'mse' scale search + per-tensor static act quant with
     leaf-parameter scale) and install a custom attention processor that
     quantizes q/k/v and the attention weights (sm_abit);
  2. PTQ4DiT tau-weighted Spearman input smoothing is computed during the FP
     calibration pass (per quantized layer);
  3. three initialization passes over calibration data: FP (smoothing) ->
     weight-quant -> weight+act-quant (scale initialization);
  4. block-wise AdaRound reconstruction (learned rounding + Lp relaxation +
     learnable activation deltas) on cached per-block inputs/outputs.

Calibration data format is identical to the TinySR pipeline:
(list of (latent, timesteps, pooled_prompt_embeds, weight_dtype), forward_fn).
The SR model runs a single fixed timestep, so the official timestep-aware
calibration reduces to single-timestep calibration (see design doc).
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.quant.ptq4dit_quant_layer import (
    QuantModule,
    UniformAffineQuantizer,
    AdaRoundQuantizer,
    lp_loss,
    StraightThrough,
)


# ---------------------------------------------------------------------------
# Args container (mirrors quant_sample.py defaults)
# ---------------------------------------------------------------------------

class PTQ4DiTArgs:
    def __init__(self, w_bits=4, a_bits=8, sm_abit=8,
                 cali_init_samples=8,
                 recon_iters=10000, recon_batch_size=4, num_split=10,
                 recon_weight=0.01, b_range=(20, 2), warmup=0.2,
                 act_lr=4e-5, p=2.0, opt_mode="mse",
                 use_tau_smooth=True):
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.sm_abit = sm_abit
        self.cali_init_samples = cali_init_samples
        self.recon_iters = recon_iters
        self.recon_batch_size = recon_batch_size
        self.num_split = num_split
        self.recon_weight = recon_weight
        self.b_range = b_range
        self.warmup = warmup
        self.act_lr = act_lr
        self.p = p
        self.opt_mode = opt_mode
        self.use_tau_smooth = use_tau_smooth
        # official defaults for the quantizers
        self.wq_params = {"n_bits": w_bits, "channel_wise": True, "scale_method": "mse"}
        self.aq_params = {"n_bits": a_bits, "symmetric": False, "channel_wise": False,
                          "scale_method": "mse", "leaf_param": True}


# ---------------------------------------------------------------------------
# Custom attention processor: JointAttnProcessor2_0 + PTQ4DiT act quantization
# ---------------------------------------------------------------------------

class PTQ4DiTAttnProcessor:
    """Attention processor mirroring JointAttnProcessor2_0 (self-attention only,
    as used by TinySR) with PTQ4DiT's q/k/v and attention-weight quantization.

    When `use_act_quant` is False it falls back to the original SDPA path,
    so the FP model behavior is bit-identical to the original processor.
    """

    def __init__(self, act_quant_params: dict, sm_abit: int = 8):
        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w["n_bits"] = sm_abit
        act_quant_params_w["always_zero"] = True
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        self.use_act_quant = False
        self.scale = None  # head scale, set from attn on first call

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.use_act_quant:
            if self.scale is None:
                self.scale = head_dim ** -0.5
            q = query * self.scale
            attn_scores = self.act_quantizer_q(q) @ self.act_quantizer_k(key).transpose(-2, -1)
            attn_scores = attn_scores.softmax(dim=-1)
            attn_scores = self.act_quantizer_w(attn_scores)
            hidden_states = attn_scores @ self.act_quantizer_v(value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states


# ---------------------------------------------------------------------------
# Layer replacement
# ---------------------------------------------------------------------------

def replace_with_ptq4dit(module: nn.Module, target_suffixes, args: PTQ4DiTArgs):
    """Replace target nn.Linear with QuantModule; install quantized processors."""
    replaced = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if "lora_" in name:
            continue
        if target_suffixes is not None and not any(name.endswith(s) for s in target_suffixes):
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = module.get_submodule(parent_name) if parent_name else module

        q_linear = QuantModule(
            org_module=child,
            weight_quant_params=copy.deepcopy(args.wq_params),
            act_quant_params=copy.deepcopy(args.aq_params),
            use_tau_smooth=args.use_tau_smooth,
        )
        setattr(parent, child_name, q_linear)
        replaced.append(name)

    # install PTQ4DiT attention processors on every transformer block
    transformer = _find_transformer(module)
    proc_count = 0
    for block in transformer.transformer_blocks:
        if hasattr(block, "attn") and hasattr(block.attn, "set_processor"):
            block.attn.set_processor(PTQ4DiTAttnProcessor(
                act_quant_params=copy.deepcopy(args.aq_params), sm_abit=args.sm_abit))
            proc_count += 1

    print(f"[PTQ4DiT] replaced {len(replaced)} nn.Linear -> QuantModule "
          f"(W{args.w_bits}A{args.a_bits}, sm_abit={args.sm_abit}), "
          f"installed {proc_count} quantized attention processors")
    return replaced


def _find_transformer(module: nn.Module):
    if hasattr(module, "transformer_blocks") and len(module.transformer_blocks) > 0:
        return module
    for m in module.modules():
        if hasattr(m, "transformer_blocks") and len(m.transformer_blocks) > 0:
            return m
    raise RuntimeError("[PTQ4DiT] could not locate transformer_blocks in module tree")


def set_ptq4dit_quant_state(module: nn.Module, weight_quant=True, act_quant=True):
    for m in module.modules():
        if isinstance(m, QuantModule):
            m.set_quant_state(weight_quant, act_quant)
        if isinstance(getattr(m, "processor", None), PTQ4DiTAttnProcessor):
            m.processor.use_act_quant = act_quant


def _set_block_quant_state(block: nn.Module, weight_quant=True, act_quant=True):
    """Set quant state for the QuantModules inside an SD3 JointTransformerBlock
    and its attention processor (the original block has no set_quant_state)."""
    for m in block.modules():
        if isinstance(m, QuantModule):
            m.set_quant_state(weight_quant, act_quant)
    attn = getattr(block, "attn", None)
    if attn is not None and isinstance(getattr(attn, "processor", None), PTQ4DiTAttnProcessor):
        attn.processor.use_act_quant = act_quant


def set_ptq4dit_running_stat(module: nn.Module, running_stat: bool):
    for m in module.modules():
        if isinstance(m, QuantModule):
            m.set_running_stat(running_stat)
        if isinstance(getattr(m, "processor", None), PTQ4DiTAttnProcessor):
            p = m.processor
            p.act_quantizer_q.running_stat = running_stat
            p.act_quantizer_k.running_stat = running_stat
            p.act_quantizer_v.running_stat = running_stat
            p.act_quantizer_w.running_stat = running_stat


def collect_ptq4dit_act_deltas(module: nn.Module):
    """Gather learnable activation deltas (weight quantizer deltas are replaced
    by AdaRound during recon; act quantizer deltas are optimized directly)."""
    params = []
    for m in module.modules():
        if isinstance(m, QuantModule):
            if m.act_quantizer.delta is not None:
                params.append(m.act_quantizer.delta)
        if isinstance(getattr(m, "processor", None), PTQ4DiTAttnProcessor):
            p = m.processor
            params += [p.act_quantizer_q.delta, p.act_quantizer_k.delta, p.act_quantizer_v.delta]
            if p.act_quantizer_w.n_bits != 16:
                params.append(p.act_quantizer_w.delta)
    return params


# ---------------------------------------------------------------------------
# Calibration: init passes (FP smoothing -> W -> WA) over the calib data
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_ptq4dit_stats(module, calib_data_list, forward_fn, args: PTQ4DiTArgs):
    """Run the three official initialization passes:
      1. FP forward  -> PTQ4DiT tau-weighted smoothing (computed lazily inside
                        QuantModule.compute_tau_smooth_scale on first call);
      2. weight-quant forward -> weight quantizer scale init (channel-wise mse);
      3. weight+act-quant forward -> activation quantizer scale init.
    """
    init_count = min(max(args.cali_init_samples, 1), len(calib_data_list))
    init_data = calib_data_list[:init_count]

    # PTQ4DiT smoothing accumulates per-sample profiles over the init samples
    # (the official computes it from its init batch; TinySR runs batch-1 forwards)
    for m in module.modules():
        if isinstance(m, QuantModule):
            m.tau_target_samples = init_count

    # pass 1: FP (triggers tau smoothing accumulation + finalize)
    set_ptq4dit_quant_state(module, weight_quant=False, act_quant=False)
    for model_input, ts, ppe, wd in init_data:
        forward_fn(model_input, ts, ppe, wd)
    # force-finalize any module that did not reach the target (fewer images)
    for m in module.modules():
        if isinstance(m, QuantModule) and not m._tau_done and m._tau_profiles:
            m.finalize_tau_smooth_scale()
    print(f"[PTQ4DiT] FP pass done (tau smoothing finalized)" )

    # pass 2: weight quant init
    set_ptq4dit_quant_state(module, weight_quant=True, act_quant=False)
    for model_input, ts, ppe, wd in init_data:
        forward_fn(model_input, ts, ppe, wd)
    print(f"[PTQ4DiT] weight quantizer init done")

    # pass 3: act quant init
    set_ptq4dit_quant_state(module, weight_quant=True, act_quant=True)
    for model_input, ts, ppe, wd in init_data:
        forward_fn(model_input, ts, ppe, wd)
    print(f"[PTQ4DiT] activation quantizer init done")

    # convert act zero_points to Parameters (as in the official flow)
    for m in module.modules():
        if isinstance(m, UniformAffineQuantizer):
            if m.delta is not None and not isinstance(m.delta, nn.Parameter):
                m.delta = nn.Parameter(m.delta)
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point.float())


# ---------------------------------------------------------------------------
# Block-wise reconstruction (ported from quant/block_recon.py + quant/utils.py)
# ---------------------------------------------------------------------------

class StopForwardException(Exception):
    pass


class _ReconBlockProxy(nn.Module):
    """Temporary proxy for a JointTransformerBlock during recon caching.

    The SD3 transformer calls blocks via `block.forward_(...)` (direct method
    call), which bypasses PyTorch's forward hooks. The proxy intercepts
    `forward_`, records the block input/output and stops the forward.
    """

    def __init__(self, real_block: nn.Module):
        super().__init__()
        self.add_module("real", real_block)
        self.record_input = True
        self.record_output = True
        self.input_store = None
        self.output_store = None

    def forward_(self, hidden_states, return_qkv=False):
        if self.record_input:
            self.input_store = hidden_states.detach()
        out = self.real.forward_(hidden_states, return_qkv=return_qkv)
        if self.record_output:
            self.output_store = out.detach()
        raise StopForwardException()
        # unreachable
        return out

    def forward(self, hidden_states, temb=None, return_qkv=False):
        # fallback for the non-initialized path (should not be hit during recon)
        return self.real(hidden_states, temb=temb, return_qkv=return_qkv)


class _DataSaverHook:
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


def _cache_block_inp_out(module, block, calib_data_list, forward_fn,
                         asym=True, act_quant=True, batch_size=4):
    """Cache the block's (input, output) over the calibration data.

    FP forward captures the reference output; with asym=True a second forward
    under the quantized network recaptures the block input (as in the official
    GetLayerInpOut / save_inp_oup_data).
    """
    device = next(module.parameters()).device
    transformer = _find_transformer(module)
    blocks = transformer.transformer_blocks
    bi = None
    for i, b in enumerate(blocks):
        if b is block:
            bi = i
            break
    if bi is None:
        raise RuntimeError("[PTQ4DiT] target block not found in transformer_blocks")

    proxy = _ReconBlockProxy(block)
    blocks[bi] = proxy

    cached_inps, cached_outs = [], []
    try:
        for i in range(0, len(calib_data_list), batch_size):
            batch = calib_data_list[i:i + batch_size]
            # gather batch latents -> single batch forward (like official batching)
            latents = torch.cat([b[0] for b in batch], dim=0)
            ts = batch[0][1]
            ppe = batch[0][2]
            wd = batch[0][3]

            # pass 1: FP forward, cache output
            proxy.record_input = True
            proxy.record_output = True
            set_ptq4dit_quant_state(module, weight_quant=False, act_quant=False)
            with torch.no_grad():
                try:
                    _run_forward(forward_fn, latents, ts, ppe, wd, device)
                except StopForwardException:
                    pass
            fp_out = proxy.output_store.cpu()
            fp_inp = proxy.input_store.cpu()

            if asym:
                # pass 2: quantized network, recapture input (keep FP output)
                proxy.record_input = True
                proxy.record_output = False
                set_ptq4dit_quant_state(module, weight_quant=True, act_quant=act_quant)
                with torch.no_grad():
                    try:
                        _run_forward(forward_fn, latents, ts, ppe, wd, device)
                    except StopForwardException:
                        pass
                q_inp = proxy.input_store.cpu()
                proxy.record_output = True
                cached_inps.append(q_inp)
            else:
                cached_inps.append(fp_inp)
            cached_outs.append(fp_out)
    finally:
        blocks[bi] = block

    set_ptq4dit_quant_state(module, weight_quant=False, act_quant=False)
    # re-enable the target block's quantization (as the official
    # GetLayerInpOut does: `self.layer.set_quant_state(True, self.act_quant)`)
    _set_block_quant_state(block, weight_quant=True, act_quant=act_quant)
    cached_inps = torch.cat(cached_inps, dim=0)
    cached_outs = torch.cat(cached_outs, dim=0)
    torch.cuda.empty_cache()
    print(f"[PTQ4DiT] cached block in={tuple(cached_inps.shape)} out={tuple(cached_outs.shape)}")
    return cached_inps, cached_outs


def _run_forward(forward_fn, latents, ts, ppe, wd, device):
    """Run the batched calibration forward (tile_sample-compatible)."""
    # forward_fn signature: (model_input, timesteps, pooled_prompt_embeds, weight_dtype)
    forward_fn(latents.to(device), ts, ppe, wd)


class LinearTempDecay:
    def __init__(self, t_max, rel_start_decay=0.2, start_b=10, end_b=2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class _LossFunction:
    def __init__(self, block, round_loss="relaxation", weight=1.0, rec_loss="mse",
                 max_count=2000, b_range=(10, 2), decay_start=0.0, warmup=0.0, p=2.0):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.temp_decay = LinearTempDecay(
            max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt):
        self.count += 1
        if self.rec_loss == "mse":
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == "cos":
            rec_loss = 1 - F.cosine_similarity(pred, tgt, dim=1).mean()
        else:
            raise ValueError(f"Not supported reconstruction loss: {self.rec_loss}")

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == "none":
            round_loss = 0
        elif self.round_loss == "relaxation":
            round_loss = 0
            for name, m in self.block.named_modules():
                if isinstance(m, QuantModule):
                    round_vals = m.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print(f"[PTQ4DiT recon] loss={float(total_loss):.4f} "
                  f"(rec={float(rec_loss):.4f}, round={float(round_loss):.4f}) "
                  f"b={b:.2f} count={self.count}")
        return total_loss


def run_ptq4dit_block_recon(module, calib_data_list, forward_fn, args: PTQ4DiTArgs):
    """Per-block AdaRound reconstruction over the calibration data.

    Ported from `block_reconstruction` (quant/block_recon.py): for each block,
    cache (quant-input, FP-output), replace weight quantizers with AdaRound,
    then optimize rounding alphas + activation deltas with the Lp relaxation
    loss. Adapted hyper-parameters: recon batch size 4 (4096-token SD3 blocks
    vs 1024-token DiT blocks) and iters 10000 by default (official 20000;
    the SR calibration set is 100 vs 1024xT samples).
    """
    transformer = _find_transformer(module)
    blocks = transformer.transformer_blocks
    num_split = args.num_split

    for bi, block in enumerate(blocks):
        block_has_quant = any(isinstance(m, QuantModule) for m in block.modules())
        if not block_has_quant:
            print(f"[PTQ4DiT] block {bi}: no quantized layers, skip recon")
            continue

        print(f"[PTQ4DiT] === block {bi} recon ===")
        set_ptq4dit_quant_state(module, weight_quant=False, act_quant=False)
        _set_block_quant_state(block, weight_quant=True, act_quant=True)

        # cache inputs/outputs (10 splits to bound memory, as official)
        b_size = max(len(calib_data_list) // num_split, 1)
        all_inps, all_outs = [], []
        for k in range(num_split):
            chunk = calib_data_list[k * b_size:(k + 1) * b_size]
            if not chunk:
                continue
            cached_inps, cached_outs = _cache_block_inp_out(
                module, block, chunk, forward_fn,
                asym=True, act_quant=True, batch_size=args.recon_batch_size)
            all_inps.append(cached_inps)
            all_outs.append(cached_outs)
        cached_inps = torch.cat(all_inps, dim=0)
        cached_outs = torch.cat(all_outs, dim=0)
        del all_inps, all_outs
        torch.cuda.empty_cache()

        # replace weight quantizers with AdaRound
        round_mode = "learned_hard_sigmoid"
        for m in block.modules():
            if isinstance(m, QuantModule):
                m.weight_quantizer = AdaRoundQuantizer(
                    uaq=m.weight_quantizer, round_mode=round_mode, weight_tensor=m.weight)
                m.weight_quantizer.soft_targets = True
        _set_block_quant_state(block, weight_quant=True, act_quant=True)

        # optimizers
        opt_params_w = []
        for m in block.modules():
            if isinstance(m, QuantModule):
                opt_params_w.append(m.weight_quantizer.alpha)
        optimizer_w = torch.optim.Adam(opt_params_w)

        opt_params_a = collect_ptq4dit_act_deltas(block)
        optimizer_a = torch.optim.Adam(opt_params_a, lr=args.act_lr)
        scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=args.recon_iters, eta_min=0.)

        loss_func = _LossFunction(
            block, round_loss="relaxation", weight=args.recon_weight,
            rec_loss=args.opt_mode, max_count=args.recon_iters,
            b_range=args.b_range, decay_start=0.0, warmup=args.warmup, p=args.p)

        device = next(module.parameters()).device
        bs = args.recon_batch_size
        per_split = args.recon_iters // num_split
        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        for k in range(num_split):
            lo, hi = k * b_size, min((k + 1) * b_size, cached_inps.shape[0])
            if lo >= hi:
                continue
            cur_x, cur_out = cached_inps[lo:hi].to(device), cached_outs[lo:hi].to(device)
            for _ in range(per_split):
                idx = torch.randperm(cur_x.size(0))[:bs]
                x_batch = cur_x[idx]
                out_batch = cur_out[idx]

                optimizer_w.zero_grad()
                optimizer_a.zero_grad()

                out_quant = block.forward_(x_batch)
                err = loss_func(out_quant, out_batch)
                err.backward()

                optimizer_w.step()
                optimizer_a.step()
                scheduler_a.step()
            del cur_x, cur_out
            torch.cuda.empty_cache()
        torch.set_grad_enabled(was_grad_enabled)

        # hard rounding
        for m in block.modules():
            if isinstance(m, QuantModule):
                m.weight_quantizer.soft_targets = False

        print(f"[PTQ4DiT] block {bi} recon done")

    set_ptq4dit_quant_state(module, weight_quant=True, act_quant=True)
    print(f"[PTQ4DiT] block recon finished; quant state ON for inference")
