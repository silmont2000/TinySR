# TinySR 对称 → 非对称量化的可行性评估

## 0. 核心约束：Nunchaku Kernel 仅支持对称量化

经过对 nunchaku 代码的三层验证，得出**确定性结论**：

| 验证层 | 证据 | 结论 |
|--------|------|------|
| Python API (`linear.py`) | `SVDQW4A4Linear` 只有 `qweight` + `wscales`，**无 `wzeros` 属性** | 数据结构不支持零点 |
| C++ 层 (`Linear.h/cpp`) | `GEMM_W4A4` 类**无 `wzeros` 成员**，`registerParams` 不含零点张量 | 内存布局不支持零点 |
| CUDA Kernel (`gemm_w4a4.cuh`) | PTX MMA 指令为 `s32.s4.s4.s32`，权重是 **signed s4**（范围 [-8,7]）；dequant 公式为 `output += q * ascale * wscale`，**无 `(q - z)` 减法** | 数学公式不支持零点 |

这意味着：**nunchaku 导出和 nunchaku 推理无法改为非对称量化**。只有 PyTorch 伪量化部分（校准 + train_quant 的 forward 推理）可以改。

---

## 1. 影响范围分析：逐文件逐行

### 1.1 可改为非对称的路径（PyTorch 伪量化）

#### A. `models/quant/quantizers.py` — `UniformAffineQuantizer`

```
状态：已支持非对称，无需修改
```

- `__init__` 已有 `symmetric` 参数（默认 `False` 表示非对称）
- `qmin`/`qmax` 属性：`symmetric=False` 时返回 `qmin=0, qmax=15`（4-bit），正确
- `calculate_qparams()`：非对称时 `scale = (max-min)/15`，`zero_point = 0 - round(min/scale)`，正确
- `_get_qparams_per_group()`：同上逻辑，正确
- `forward()`：非对称时 clamp 到 `[0, 15]`，正确

#### B. `models/quant/inference.py:40-53` — `build_layer_replacement_kwargs()`

```
当前：硬编码 "symmetric": True（权重）+ "symmetric": True（激活）
需改：→ False
```

```python
# 第 43-52 行
weight_quant_kwargs = {
    "bits": w_bits, "symmetric": False,  # ← 改这里
    ...
}
act_quant_kwargs = {
    "bits": a_bits, "symmetric": False,  # ← 改这里
    ...
}
```

**建议**：不要硬编码，新增 CLI 参数 `--quant_symmetric`（默认 `True`，保持向后兼容）

#### C. `models/quant/calibrate.py` — 校准逻辑

```
状态：已正确传递 symmetric 参数，无需核心逻辑修改
```

- `_eval_quant_error()`：通过参数 `act_symmetric` 控制，调用方从 `aq.quantizer.symmetric` 读取（`calibrate.py:137`），已正确
- `search_alpha()`：同上，通过 `act_sym` 参数传递（`calibrate.py:155`），已正确
- `calibrate_one_layer()` → `build_branch()`：`decompose_svd_branch(symmetric=self.quantizer.symmetric)`，已正确传递（`components.py:312`）

**仅需验证一处**：`_eval_quant_error` 中 weight quantization 的 symmetric 参数从 `weight_quantizer.quantizer.symmetric` 读取（`calibrate.py:40`），确认已正确。

#### D. `models/quant/ops.py` — 量化算子

```
状态：所有函数已支持 symmetric 参数，无需修改
```

- `affine_fake_quant_weight(symmetric=...)` ✓
- `gptq_per_group_int4(symmetric=...)` ✓
- `gptq_quantize_linear_weight(symmetric=...)` ✓
- `fake_quant_activation(symmetric=...)` ✓

#### E. `models/quant/components.py` — `LowRankAffineQuantComponent`

```
状态：基本已支持，需检查一处
```

- `decompose_svd_branch(symmetric=...)`：已透传 ✓
- `build_branch()` → `decompose_svd_branch(symmetric=self.quantizer.symmetric)`：已正确 ✓
- **`_build_nunchaku_residual()`（components.py:331-352）**：硬编码 `symmetric=True`，但此方法仅用于 nunchaku 对齐，nunchaku 本身不支持非对称 → **不改**

#### F. `models/quant/layers.py` — `QuantLinearW4A4.forward()`

```
状态：无 symmetric 硬编码，无需修改
```

Forward pass 无直接量化逻辑，仅调用 `self.act_quantizer(x)` 和 `self.weight_quantizer(weight)`，对称性由 quantizer 内部决定 ✓

---

### 1.2 不可改为非对称的路径（Nunchaku 相关）

#### G. `train_quant.py:120-253` — `save_nunchaku_safetensors()`

```
状态：硬编码 symmetric=True，且不可改
```

| 行号 | 位置 | 硬编码 | 可否改 |
|------|------|--------|--------|
| 165 | `assert in_features % group_size == 0` | - | - |
| 177 | `gptq_per_group_int4(..., symmetric=True, ...)` | `symmetric=True` | **不可改**（nunchaku kernel 不支持不对称） |
| 183-184 | Minmax 路径：`abs().amax(...) / 7.0` + `clamp_(-8, 7)` | 对称公式 | **不可改**（同上） |

#### H. `test/tinysd3_nunchaku_w4a4.py:41-60` — `quantize_linear_weight_w4a4()`

```
状态：硬编码对称公式，且不可改
```

```python
# 第 52-54 行
scale = weight_groups.abs().amax(dim=-1).clamp_min(eps) / 7.0
qweight = torch.round(weight_groups / scale.unsqueeze(-1)).clamp_(-8, 7).to(torch.int32)
```

这两行是 nunchaku kernel 前向所需的数据格式，**无法改为非对称**。

#### I. `models/quant/layers.py:172-198` — `_forward_nunchaku_aligned()`

```
状态：硬编码对称逻辑，但可改为仅在 symmetric 模式时启用
```

- 第 176-181 行：动态 per-group 激活量化使用对称公式（`abs().amax(...) / 7.0`）
- 第 184 行：使用 `_nunchaku_residual`（对称 GPTQ 残差）

**处理方式**：当 `symmetric=False` 时，不启用 nunchaku-aligned 模式（因为 nunchaku 本身也不支持非对称）。

---

### 1.3 需要新增 CLI 参数

在 `train/train_quant.py` 的 `parse_args()` 中新增：

```python
parser.add_argument("--quant_symmetric", action="store_true", default=True,
                    help="Use symmetric quantization (default). "
                         "Set --no-quant_symmetric for asymmetric.")
```

可选扩展为权重和激活分别控制：
```python
parser.add_argument("--w_symmetric", action="store_true", default=True)
parser.add_argument("--a_symmetric", action="store_true", default=True)
```

---

## 2. 修改清单

### 必须改（2 处）

| # | 文件 | 位置 | 修改内容 |
|---|------|------|----------|
| 1 | `train/train_quant.py` | `parse_args()` | 新增 `--quant_symmetric` / `--w_symmetric` / `--a_symmetric` 参数 |
| 2 | `models/quant/inference.py` | `build_layer_replacement_kwargs():43-52` | `symmetric` 改为从参数读取，默认保持 `True` |

### 需验证（3 处）

| # | 文件 | 检查项 |
|---|------|--------|
| 3 | `models/quant/calibrate.py:40` | 确认 `_eval_quant_error` 中 weight 的 `symmetric` 参数从 `weight_quantizer.quantizer.symmetric` 正确读取 |
| 4 | `models/quant/components.py:312` | 确认 `decompose_svd_branch` 收到正确的 `symmetric` 值 |
| 5 | `models/quant/quantizers.py` | per-group 非对称量化路径（`_get_qparams_per_group`、`forward`）已有但未经测试 |

### 不改（Nunchaku 路径，3 处）

| # | 文件 | 原因 |
|---|------|------|
| 6 | `train/train_quant.py:save_nunchaku_safetensors()` | Nunchaku kernel 不支持非对称 |
| 7 | `test/tinysd3_nunchaku_w4a4.py:quantize_linear_weight_w4a4()` | 同上 |
| 8 | `models/quant/layers.py:_forward_nunchaku_aligned()` | 同上（非对称时自动跳过 nunchaku-aligned 模式） |

### 可能需改（1 处）

| # | 文件 | 内容 |
|---|------|------|
| 9 | `models/quant/components.py:_build_nunchaku_residual()` | 非对称时跳过此方法（nunchaku 残差仅用于 nunchaku 对齐，非对称下无意义） |

---

## 3. 可行性结论

| 路径 | 可行性 | 工时 | 说明 |
|------|--------|------|------|
| **PyTorch 校准** | ✅ 可行 | 0.5 天 | 框架已支持，仅需暴露参数 + 验证 |
| **PyTorch 推理（train_quant）** | ✅ 可行 | 0.5 天 | 同上 |
| **Nunchaku 导出** | ❌ 不可行 | — | CUDA kernel 数学公式不支持零点 |
| **Nunchaku 推理** | ❌ 不可行 | — | 同上 |

**总工时**：1 天（含验证）。

---

## 4. 替代方案：不对称校准 + 对称导出

如果目标是利用非对称量化的校准优势同时保持 nunchaku 兼容性，可以：

1. **校准阶段**使用非对称量化（可能得到更好的 smooth_alpha、更优的 SVD 分解）
2. **导出阶段**将权重"对称化"：`w_sym_quant = clamp(round((w - min) / ((max - min) / 15) * 7 / 7.5 + ...))`

但这是一种**有损转换**，且需要额外验证精度损失。不推荐作为主路径。

**更务实的做法**：将非对称的方案仅用于 PyTorch 伪量化对比实验，在和其他方法（DiTAS、ViDiT-Q等）做公平对比时使用，而生产环境的 nunchaku 导出保持对称量化。

---

## 5. 对其他五种方法的启示

| 方法 | 量化类型 | TinySR 适配后如何处理 Nunchaku |
|------|----------|-------------------------------|
| PTQ4DiT | 非对称 W/A | 伪量化对比用非对称；nunchaku 导出需降级为对称 |
| Q-DiT | 非对称（默认） | 同上 |
| ViDiT-Q | 激活对称、权重非对称（默认） | 混合模式，nunchaku 导出需统一为对称 |
| **DiTAS** | **非对称 W/A** | **这是五种方法中与当前对称 TinySR 差异最大的点之一** |
| DVD-Quant | 权重非对称、激活对称 | 需降级导出 |

这进一步验证了前一个文档中的结论：DiTAS 的非对称量化是关键的差异化对比维度。
