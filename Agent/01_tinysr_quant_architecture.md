# TinySR 量化架构与逻辑详解

## 1. TinySR 整体项目结构

TinySR 是一个基于 Stable Diffusion 3 (SD3) DiT（Diffusion Transformer）架构的超分辨率模型。核心流水线：

```
输入低分辨率图像 → VAE 编码 → DiT Transformer（去噪预测）→ VAE 解码 → 高分辨率输出
```

- **模型主体**：`TinySD3Transformer2DModel`（`models/tinysr/tinysd3.py`），由多个 `JointTransformerBlock` 组成
- **VAE**：`AutoencoderTiny`（`models/vae/autoencoder_tiny.py`）
- **LoRA**：可选的低秩适配器微调
- **骨干网络**：prune-12-merge-tinysr（剪枝+合并后的 checkpoint）

---

## 2. 当前量化方案：SVDQ-W4A4

### 2.1 概述

方案全称为 **SVDQ-W4A4**（SVD + Quantization），是一个 **Weight 4-bit + Activation 4-bit** 的混合精度量化方案。

核心思想：
1. 使用 **SVD 低秩分解** 提取权重的主要方向，用低秩分支（A×B）近似
2. 残差部分（原始权重 - 低秩近似）做 **GPTQ 量化**（4-bit int）
3. 引入 **Smooth Quantization** 机制平衡权重和激活的量化难度
4. 推理时：quantized_residual @ quantized_act + low_rank_branch(act)

### 2.2 核心模块：`QuantLinearW4A4`

文件：`models/quant/layers.py:115-202`

每个被量化的 `nn.Linear` 层被替换为 `QuantLinearW4A4`，包含：

| 组件 | 类型 | 职责 |
|------|------|------|
| `weight` | `nn.Parameter` | 原始 FP 权重（不变） |
| `weight_quantizer` | `LowRankAffineQuantComponent` | 权重量化（含 SVD 分支 + smooth_scale） |
| `act_quantizer` | `LowRankAffineQuantComponent` | 激活量化 |

前向传播逻辑（`forward`，`layers.py:150-170`）：

```
1. x = x / smooth_scale                 # 平滑激活
2. w = weight * smooth_scale             # 平滑权重
3. x_q = act_quantizer(x)               # 伪量化激活（4-bit per-group）
4. w_q = weight_quantizer(w)            # 已冻结的残差量化权重
5. out = F.linear(x_q, w_q)             # 主分支
6. out += weight_quantizer.branch_forward(x)  # SVD 低秩分支
```

### 2.3 核心组件：`LowRankAffineQuantComponent`

文件：`models/quant/components.py:178-398`

这是整个量化方案的灵魂组件，包含：

| 属性 | 说明 |
|------|------|
| `quantizer` | `UniformAffineQuantizer`，基础均匀量化器 |
| `rank` | SVD 低秩分解的秩（默认 32） |
| `smooth_alpha` | SmoothQuant 的迁移系数 α ∈ [0, 1] |
| `smooth_scale` | 平滑缩放向量 s ∈ R^(in_features) |
| `branch` | `LowRankBranch`，SVD 低秩分支（A ∈ R^(rank×in), B ∈ R^(out×rank)） |
| `residual` | 量化后的残差权重（冻结后使用） |
| `_nunchaku_residual` | Nunchaku 对齐的 per-group-per-channel GPTQ 残差 |
| `input_cache` | 校准输入缓存（用于 GPTQ） |
| `raw_input_cache` | 原始输入缓存（用于 nunchaku 导出时的 GPTQ） |
| `act_absmax` | 激活的逐通道绝对最大值 |

**Smooth Quantization 公式**（`calibrate.py:16-19`）：

```
s = act_absmax^α / weight_absmax^(1-α)
```

- α→1：s ~ act_absmax，放大权重、缩小激活，将量化难度从激活转移到权重
- α→0：s ~ 1/weight_absmax
- α=0.5（默认）：两者平衡

### 2.4 基础量化器：`UniformAffineQuantizer`

文件：`models/quant/quantizers.py:38-164`

支持：
- **对称/非对称量化**（4-bit: qmin=-8, qmax=7）
- **逐通道/逐张量**量化
- **Per-group 量化**（`group_size=64`），用于激活
- 在线计算 scale/zero_point（从 MinMaxObserver 收集的 min/max）

### 2.5 SVD 低秩分解

文件：`models/quant/components.py:11-75`（`decompose_svd_branch`）

```
1. smoothed_weight = weight * smooth_scale
2. U, S, Vh = SVD(smoothed_weight)
3. branch_A = Vh[:rank]           # (rank, in_features)
4. branch_B = U[:, :rank] * S[:rank]  # (out_features, rank)
5. low_rank = branch_B @ branch_A
6. residual = smoothed_weight - low_rank
7. residual_quantized = GPTQ(residual, calibration_inputs)
```

支持**迭代精炼**（`num_svd_iterations` > 0）：
- 交替执行：SVD(weight - quantized_residual) → 重量化 → 检查误差是否下降
- 不下降时 early stop

---

## 3. 量化校准流水线（Calibration Pipeline）

文件：`models/quant/calibrate.py`、`models/quant/inference.py`

### 3.1 入口函数

`train_quant.py:main()` → `calibrate_w4a4()` → `run_calibration()`

### 3.2 Phase 1：激活统计收集

```
set_quant_enabled(False)   # 关闭所有量化
set_observer_enabled(True)  # 开启观察者模式

for each calibration image:
    forward pass (FP16)
    → 收集 act_absmax（逐通道激活绝对值最大值）
    → 收集 input_cache（用于后续 GPTQ）
```

### 3.3 Phase 2：逐层校准

三种模式：

| 模式 | CLI 参数 | 说明 |
|------|---------|------|
| 固定 α | `--svdq_smooth_alpha 0.5`（默认） | 所有层使用相同 α，不做误差计算 |
| 网格搜索 | `--search_smooth_alpha grid` | 每层搜索 α ∈ {0.0, 0.1, ..., 1.0}，选最小重构误差 |
| 级联冻结 | `--search_smooth_alpha cascade` | 逐层冻结，后续层能感知前层的量化影响 |
| 报告加载 | `--load_smooth_alpha_report` | 从历史 JSON 加载逐层 α（最高优先级） |

**逐层校准步骤**（`calibrate_one_layer`，`calibrate.py:203-230`）：

1. **解析 α**：
   - 优先级：override > search > default
   - 计算 `smooth_scale = act_absmax^α / weight_absmax^(1-α)`

2. **冻结 weight_quantizer**：
   - 计算量化 scale（从观察到的 min/max）
   - `build_branch()`：SVD → 低秩分支 + GPTQ 量化残差

3. **冻结 act_quantizer**：
   - 根据收集的激活统计计算量化 scale

4. **GPTQ 量化细节**（`ops.py:118-222`）：
   - 使用校准输入构建 Hessian 矩阵 `H = X^T X`
   - 按 Hessian 对角线重要性排序列
   - 逐列量化，每次量化后将误差传播到后续列
   - 支持 block_size=128 分块处理大矩阵
   - 支持 per-group 模式（group_size=64）

### 3.4 级联模式特殊逻辑

`calibrate_all_layers_cascade`（`calibrate.py:270-334`）：
- 逐层处理，每层处理前：
  1. 重置该层的 input_cache
  2. 用少量校准图片（默认 4 张）做前向传播，重新收集 **后量化激活**
  3. 保持 Phase 1 的大样本 act_absmax 不变（用于准确的 α 搜索）
- `ff.net.2` 层使用 `gate_mlp` 加权 MSE，因为 `AdaLayerNormZero` 的逐通道 gating 使得某些通道更重要

---

## 4. 量化范围（Quant Scope）

文件：`models/quant/layers.py:8-46`

```python
FFN_SUFFIXES = ["ff.net.0.proj.base_layer", "ff.net.2.base_layer",
                "ff.net.0.proj", "ff.net.2"]
ATTN_SUFFIXES = ["attn.to_q.base_layer", "attn.to_k.base_layer",
                 "attn.to_v.base_layer", "attn.to_out.0.base_layer",
                 "attn.to_q", "attn.to_k", "attn.to_v"]
```

- `ffn_only`：仅量化 FFN 中的 Linear 层
- `attn_only`：仅量化 Attention 中的 Linear 层（可选混合部分 FFN blocks）
- `dit_full`：两者都量化

### 4.1 当前实际使用的量化配置

项目实践中主要使用以下配置组合：

- **量化范围**：`attn_only` + 指定 FFN blocks `1,6,9,10,11`
  - 即 Attention 的 `to_q`、`to_k`、`to_v` 全部量化
  - FFN 仅量化 blocks 1, 6, 9, 10, 11 的 `ff.net.0.proj` 和 `ff.net.2`
  - CLI 示例：`--quant_scope attn_only --quant_ffn_blocks "1,6,9,10,11"`
- **α 策略**：主要使用——固定 α（`--svdq_smooth_alpha`）或级联冻结（`--search_smooth_alpha cascade`）
- **校准数据**：使用 DRealSRVal_crop128 的 test_LR 图片集

---

## 4.2 混合量化配置说明

`attn_only` + 指定 FFN blocks 的设计意图：
- Attention 层的量化压缩效果显著（Attention 中有大量 Linear 层）
- FFN 层对精度更敏感，因此只选择部分关键 blocks 量化，其余保持 FP16
- blocks 1,6,9,10,11 是经验选择——既保证压缩比，又不显著损害 SR 质量

`parse_ffn_blocks`（`layers.py:76-112`）支持灵活语法：
```
"1,6,9,10,11"           → 全部 "both"（即 ff.net.0.proj + ff.net.2）
"1.up,6,9.down"         → 按 mode 选择（up=仅 ff.net.0.proj, down=仅 ff.net.2）
"1u,6d"                 → 简写语法
```

---

## 5. Nunchaku 导出

文件：`train_quant.py:120-253`（`save_nunchaku_safetensors`）

导出为 Nunchaku 引擎兼容的 safetensors 格式：

1. **逐层处理每个 `QuantLinearW4A4`**：
   - 反平滑残差：`residual_for_nunchaku = (weight * s - branch_B @ branch_A) / s`
   - Per-group int4 量化残差（group_size=64）
   - 优先用 GPTQ（如果 raw_input_cache 有数据），否则用 minmax
   - 打包 qweight（int32→int8 packed 格式）
   - 打包 wscales（nunchaku 布局）
   - 打包 proj_down（`branch_A / smooth_scale`）、proj_up（`branch_B`）

2. **输出 key 格式**：
   - `<layer>.qweight`：打包的 int4 量化权重
   - `<layer>.wscales`：打包的 per-group scales
   - `<layer>.proj_down`：低秩投影降维
   - `<layer>.proj_up`：低秩投影升维
   - `<layer>.smooth_factor`：平滑因子（当前导出为全 1）
   - `<layer>.smooth_factor_orig`：同上
   - `<layer>.hadamard_rotated`（如果存在）
   - `<layer>.bias`（如果存在）

---

## 6. Nunchaku 推理

文件：`test/test_nunchaku_inference.py`

### 6.1 模型加载流程

1. 加载合并后的骨干模型（`TinySD3Transformer2DModel.from_pretrained`）
2. 加载 VAE
3. **替换 `nn.Linear` → `NunchakuSVDQLinear`**（`replace_linear_with_nunchaku`）
   - 根据 `quant_scope` + `target_suffixes` 选择目标层
4. 加载 nunchaku safetensors（`load_nunchaku_state`）
5. 转为 eval 模式

### 6.2 Tile Sampling 推理

- 大图分 tile（默认 64×64，overlap=8）
- 每个 tile 独立前向传播
- 用 Gaussian 权重融合重叠区域

### 6.3 Nunchaku-aligned 验证模式

`train_quant.py` 支持 `--align_nunchaku_inference`，使 PyTorch 前向与 Nunchaku CUDA kernel 输出完全一致：
- 动态 per-group int4 激活量化（`QuantLinearW4A4._forward_nunchaku_aligned`）
- Per-group-per-channel GPTQ 残差权重（`_nunchaku_residual`）
- 反平滑的 LoRA 投影（`proj_down / smooth_scale`）

---

## 7. 输出与分析

### 7.1 输出
- SR 图像（存放在 `<output_dir>/`）
- `w4a4_report.json`：包含 args、量化层信息、量化元数据、推理耗时
- `nunchaku.safetensors`：nunchaku 格式的量化权重
- `merged_backbone/`：合并后的 FP 骨干权重（config.json + safetensors）
- `activation_error_report.json`（可选）：逐层激活量化误差统计

### 7.2 激活误差分析（`ActivationErrorAnalyzer`）
文件：`models/quant/analysis.py`

使用 hook 机制捕获每层激活量化的输入输出，计算：
- MSE、MAE、Max Abs Error
- SNR（信噪比）
- 峰度（Kurtosis）、异常值比率

---

## 8. 关键参数总结

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `w_bits` | 4 | 权重量化位宽 |
| `a_bits` | 4 | 激活量化位宽 |
| `svdq_rank` | 32 | SVD 低秩分解的秩 |
| `svdq_smooth_alpha` | 0.5 | SmoothQuant 迁移系数 |
| `act_group_size` | 64 | 激活 per-group 量化粒度 |
| `weight_group_size` | -1 | 权重 per-group 量化粒度（-1 = per-channel） |
| `svdq_iterations` | 0 | SVD 迭代精炼次数 |
| `calib_images` | 8 | 校准图片数量 |
| `quant_scope` | ffn_only | 量化范围 |
| `search_smooth_alpha` | None | α 搜索模式（grid/cascade） |

---

## 9. 代码文件索引

| 文件 | 功能 |
|------|------|
| `train/train_quant.py` | 量化训练/推理主入口 |
| `test/test_nunchaku_inference.py` | Nunchaku 加速推理 |
| `models/quant/layers.py` | `QuantLinearW4A4` 层定义、层替换、后缀规则 |
| `models/quant/components.py` | `LowRankAffineQuantComponent`、`LowRankBranch`、SVD 分解 |
| `models/quant/quantizers.py` | `UniformAffineQuantizer`、`MinMaxObserver` |
| `models/quant/calibrate.py` | 校准流水线、α 搜索、逐层冻结 |
| `models/quant/ops.py` | GPTQ 量化算子、伪量化函数 |
| `models/quant/inference.py` | 量化层替换入口、校准入口封装 |
| `models/quant/tiler.py` | 潜在空间 Tile 采样 |
| `models/quant/analysis.py` | 激活量化误差分析 |
| `models/pipeline.py` | 通用流水线：模型加载、图像预处理、VAE 编解码、推理 |
