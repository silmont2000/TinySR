"""Q-DiT quantized layers for TinySR (SD3-based MMDiT backbone).

Faithful port of the official Q-DiT implementation
(https://github.com/Juanerx/Q-DiT, CVPR 2025) onto TinySR's layer scheme:

  - qdit/quant.py          -> quantize_tensor / quantize_tensor_channel_group /
                              Quantizer (dynamic per-token activation)
  - qdit/qLinearLayer.py   -> QLinearLayer (weight quantization)
  - qdit/gptq.py           -> Quantizer_GPTQ / GPTQ (Hessian-based refinement)

Adaptations (documented in agent/qdit_ptq4dit_integration_design.md):
  * TDC (temporal discrepancy-aware calibration) reduces to single-timestep
    calibration: the SR model runs a fixed timestep (1000.0), so the calibration
    data has no time dimension.
  * BEM (block-wise error minimization) is the per-block sequential GPTQ loop,
    identical to the official `quantize_model_gptq`.
  * The vanilla-DiT fused qkv is split into to_q/to_k/to_v (SD3 joint attention);
    dynamic per-token activation quantization of the shared input is applied
    per layer, which is functionally equivalent to a single input_quant.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ---------------------------------------------------------------------------
# Weight quantization primitives (ported from qdit/quant.py)
# ---------------------------------------------------------------------------

def lp_loss(pred, tgt, p=2.0):
    x = (pred - tgt).abs().pow(p)
    y = torch.flatten(x, 1)
    return y.mean(1, keepdim=True)


@torch.no_grad()
def quantize_tensor(w, n_bits, group_size, sym, clip_ratio=1.0,
                    quant_type="int", quant_method="max"):
    """Per-channel / per-group weight quantization (dequantized output).

    Mirrors the official `qdit.quant.quantize_tensor`:
      - group_size == 0  -> per-channel quantization
      - group_size > 0   -> the last dim is split into groups of `group_size`
      - quant_method 'max' -> min/max scale
      - quant_method 'mse' -> L_p grid search over clip ratios (LAPQ-style)
    """
    savedShape = w.shape
    w = w.squeeze()
    if not w.is_contiguous():
        w = w.contiguous()
    if group_size > 0:
        assert w.shape[-1] % group_size == 0, (
            f"weight last dim {w.shape[-1]} not divisible by group_size {group_size}")
        w = w.reshape(-1, group_size)
    assert w.dim() == 2, "Weight format should be: [num_groups, group_size]"
    assert n_bits < 16
    assert quant_type == "int", "Options should be in [int, fp]"

    if quant_method == "max":
        if sym:
            w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        else:
            w_max = w.amax(dim=-1, keepdim=True)
            w_min = w.amin(dim=-1, keepdim=True)

        if sym:
            q_max = (2 ** (n_bits - 1) - 1)
            q_min = (-2 ** (n_bits - 1))
            if clip_ratio < 1.0:
                w_max = w_max * clip_ratio
            scales = w_max / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2 ** n_bits - 1)
            q_min = (0)
            if clip_ratio < 1.0:
                w_max *= clip_ratio
                w_min *= clip_ratio
            scales = (w_max - w_min).clamp(min=1e-5) / q_max
            base = torch.round(-w_min / scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales

    elif quant_method == "mse":
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        w_absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        best_score = torch.zeros_like(w_max) + (1e10)
        best_min = w_min.clone()
        best_max = w_max.clone()
        best_absmax = w_absmax.clone()
        for i in range(100):
            if sym:
                new_max = w_absmax * (1.0 - (i * 0.001))
                q_max = (2 ** (n_bits - 1) - 1)
                q_min = (-2 ** (n_bits - 1))
                scales = new_max / q_max
                base = torch.zeros_like(scales)
            else:
                new_max = w_max * (1.0 - (i * 0.001))
                new_min = w_min * (1.0 - (i * 0.001))
                q_max = (2 ** n_bits - 1)
                q_min = (0)
                scales = (new_max - new_min).clamp(min=1e-5) / q_max
                base = torch.round(-new_min / scales).clamp_(min=q_min, max=q_max)
            w_q = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
            score = lp_loss(w, w_q, p=2.4)
            if sym:
                best_absmax = torch.where(score < best_score, new_max, best_absmax)
            else:
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(best_score, score)
        if sym:
            q_max = (2 ** (n_bits - 1) - 1)
            q_min = (-2 ** (n_bits - 1))
            scales = best_absmax / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2 ** n_bits - 1)
            q_min = (0)
            scales = (best_max - best_min).clamp(min=1e-5) / q_max
            base = torch.round(-best_min / scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales

    else:
        raise NotImplementedError

    return w.reshape(savedShape)


@torch.no_grad()
def quantize_tensor_channel_group(W, n_bits, group_size, sym, channel_group=1,
                                  clip_ratio=1.0, quant_type="int", quant_method="max"):
    """Wrapper: continuous `channel_group` channels share one quantization setup."""
    assert W.is_contiguous(), "Input tensor is not contiguous"
    assert n_bits < 16

    if group_size > 0:
        assert W.shape[-1] % group_size == 0, (
            f"weight last dim {W.shape[-1]} not divisible by group_size {group_size}")

    if group_size == 0:
        W = quantize_tensor(W, n_bits=n_bits, group_size=0, sym=sym,
                            clip_ratio=clip_ratio, quant_type=quant_type,
                            quant_method=quant_method)
    else:
        for i1 in range(0, W.shape[1], group_size):
            i2 = min(i1 + group_size, W.shape[1])
            w = W[:, i1:i2]
            if channel_group > 1:
                w = w.reshape(int(W.shape[0] / channel_group), -1).contiguous()
            w = quantize_tensor(w, n_bits=n_bits, group_size=0, sym=sym,
                                clip_ratio=clip_ratio, quant_type=quant_type,
                                quant_method=quant_method)
            if channel_group > 1:
                w = w.reshape(-1, group_size)
            W[:, i1:i2] = w
    return W.contiguous()


# ---------------------------------------------------------------------------
# Activation quantization (ported from qdit/quant.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_activation_wrapper(x, n_bits, group_size, sym, clip_ratio=1.0,
                                quant_type="int"):
    """Dynamic per-token activation quantization (dequantized output)."""
    if n_bits >= 16:
        return x
    savedShape = x.shape
    x = x.reshape(-1, savedShape[-1])
    if group_size > 0:
        assert savedShape[-1] % group_size == 0, (
            f"act last dim {savedShape[-1]} not divisible by group_size {group_size}")
    x = quantize_tensor(x, n_bits=n_bits, group_size=group_size, sym=sym,
                        clip_ratio=clip_ratio, quant_type=quant_type,
                        quant_method="max")
    return x.view(savedShape)


@torch.no_grad()
def quantize_attn_qkv_wrapper(w, n_bits, head_dim, sym=False, clip_ratio=1.0,
                              quant_type="int"):
    """Quantize q/k/v on the head dimension (per-head, dynamic per-token)."""
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)
    w = quantize_tensor(w, n_bits=n_bits, group_size=0, sym=sym,
                        clip_ratio=clip_ratio, quant_type=quant_type,
                        quant_method="max")
    return w.view(saved_shape)


class Quantizer(nn.Module):
    """Activation quantizer: dynamic by default; static when scales provided.

    Ported from `qdit.quant.Quantizer`. The static path is optional and kept
    for API compatibility (the official default config uses dynamic scales).
    """

    def __init__(self, act_fn=None):
        super().__init__()
        self.register_buffer("scales", None)
        self.act_quant = act_fn if act_fn is not None else (lambda x: x)

    def configure(self, act_fn, scales):
        self.act_quant = act_fn
        if scales is not None:
            self.scales = scales

    def forward(self, hidden_states):
        if self.scales is not None:
            savedShape = hidden_states.shape
            hidden_states = hidden_states.view(-1, savedShape[-1])
            selected_states = hidden_states.clone()
            B, N, C = savedShape
            scales = self.scales[0].unsqueeze(0).repeat(B * N, 1)
            base = self.scales[1].unsqueeze(0).repeat(B * N, 1)
            selected_states = (torch.clamp(torch.round(selected_states / scales) + base,
                                           self.q_min, self.q_max) - base) * scales
            hidden_states = selected_states.view(savedShape)
            return hidden_states
        return self.act_quant(hidden_states)


# ---------------------------------------------------------------------------
# QLinearLayer (ported from qdit/qLinearLayer.py)
# ---------------------------------------------------------------------------

def find_qlinear_layers(module, name=""):
    res = {}
    if isinstance(module, QLinearLayer) and module.enable_quant:
        return {name: module}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + "." + name1 if name != "" else name1))
    return res


class QLinearLayer(nn.Module):
    """Linear layer whose weight is quantized in-place (dequantized values)."""

    def __init__(self, originalLayer, args, enable_quant=True):
        super().__init__()
        self.args = args
        self.register_buffer("weight", originalLayer.weight.data)
        self.enable_quant = enable_quant
        if originalLayer.bias is not None:
            self.register_buffer("bias", originalLayer.bias.data)
        else:
            self.bias = None
        self.quantized = False

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return
        self.weight = quantize_tensor_channel_group(
            self.weight.clone(),
            n_bits=self.args.wbits,
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            quant_type=self.args.quant_type,
            quant_method=self.args.quant_method,
        )
        self.quantized = True

    def extra_repr(self):
        return (f"wbit={self.args.wbits}, sym={self.args.w_sym}, "
                f"group_size={self.args.weight_group_size}, quantized={self.quantized}")


class QDiTQuantLinear(QLinearLayer):
    """Q-DiT quantized linear with dynamic per-token input activation quantizer.

    Activation placement mirrors Q-DiT's `input_quant` / `act_quant` points,
    applied per quantized layer (see design doc).
    """

    def __init__(self, originalLayer, args, enable_quant=True):
        super().__init__(originalLayer, args, enable_quant=enable_quant)
        self.input_quant = Quantizer()
        self.input_quant.configure(
            partial(quantize_activation_wrapper,
                    n_bits=args.abits,
                    group_size=args.act_group_size,
                    sym=args.a_sym,
                    clip_ratio=args.a_clip_ratio,
                    quant_type=args.quant_type),
            None,
        )

    def forward(self, x):
        x = self.input_quant(x)
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# GPTQ weight refinement (ported from qdit/gptq.py)
# ---------------------------------------------------------------------------

class Quantizer_GPTQ(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer_GPTQ, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, perchannel=False, channel_group=1, sym=True,
                  mse=False, norm=2.4, grid=100, maxshrink=.8,
                  clip_ratio=1.0, trits=False, quant_type="int"):
        if quant_type == "int":
            self.maxq = torch.tensor(2 ** bits - 1)
        else:
            assert quant_type == "fp", "Currently only support [int, fp]."
            self.maxq = torch.tensor(2 * 12.0, dtype=torch.float32)
        self.perchannel = perchannel
        self.channel_group = channel_group
        if self.channel_group > 1:
            assert self.perchannel is True, "set perchannel to True when using multilple channel group"
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.clip_ratio = clip_ratio
        self.quant_type = quant_type
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.channel_group > 1:
                    x = x.reshape(int(shape[0] / self.channel_group), -1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) * self.clip_ratio / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize_gptq(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq,
                                  self.channel_group, self.quant_type)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize_gptq(x, self.scale, self.zero, self.maxq,
                                 self.channel_group, self.quant_type)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


@torch.no_grad()
def quantize_gptq(x, scale, zero, maxq, channel_group, quant_type="int"):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    shape = x.shape
    if channel_group > 1:
        assert len(shape) == 2, "only support 2D input when using multilple channel group"
        shape = x.shape
        x = x.reshape((int(x.shape[0] / channel_group), -1))
    if quant_type == "int":
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        q = scale * (q - zero)
    else:
        raise NotImplementedError("fp quantization not supported in TinySR port")
    return q.reshape(shape)


class GPTQ:
    """Hessian-based per-column weight refinement (ported from qdit/gptq.py)."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.n_nonout = W.shape[1]
        del W

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, (nn.Linear, QLinearLayer)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        assert actorder is False, "actorder not supported in TinySR port"
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W[:, :self.n_nonout], weight=True)

        H = self.H.clone()
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.n_nonout, blocksize):
            i2 = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize > 0:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i):min((i1 + i + groupsize), self.n_nonout)],
                            weight=True)
                q = quantize_gptq(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero,
                    self.quantizer.maxq, self.quantizer.channel_group,
                    self.quantizer.quant_type
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        del H
        del Losses
        del W

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
