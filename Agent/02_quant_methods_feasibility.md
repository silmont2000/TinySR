# 五种 DiT 量化方法对 TinySR 的适配可行性评估

## 0. TinySR 架构关键特征

在进行适配评估前，先明确 TinySR 与各方法原始目标架构的差异：

| 特征 | TinySR (SD3 系) | 原始 DiT (Meta) |
|------|----------------|-----------------|
| Block 类型 | `JointTransformerBlock` | `DiTBlock` |
| Block 数量 | 12（prune后）或 18 | 28（XL）/ 24（L）/ 12（B） |
| 隐藏维度 | 1152（18 heads × 64） | 1152（XL） |
| 注意力 Q/K/V | `attn.to_q`, `attn.to_k`, `attn.to_v` | `attn.qkv`（融合） |
| 注意力输出 | `attn.to_out[0]`（ModuleList内） | `attn.proj` |
| FFN 结构 | `ff.net[0].proj`（GELU内）+ `ff.net[2]` | `mlp.fc1` + `mlp.fc2` |
| adaLN 调制 | `gate_msa/scale_msa/shift_msa` 等 buffer（首轮后冻结） | `adaLN_modulation` 每次 forward 计算 |
| 文本交叉注意力 | 不存在（单流，仅图像） | 不存在 |
| LoRA 包装 | 可选，产生 `base_layer` 包装 | 无 |

**关键差异**：TinySR 使用 SD3 架构，与所有五种方法的目标架构（原始 Meta DiT）在模块命名、block 结构、调制机制上均不同。**所有方法都需要做架构适配**。

---

## 1. 各方法概览与适配评估

---

### 1.1 PTQ4DiT（NeurIPS 2024）

| 维度 | 详情 |
|------|------|
| **论文** | PTQ4DiT: Post-training Quantization for Diffusion Transformers |
| **量化位宽** | W8A8 / W4A8（论文主要评估） |
| **核心算法** | Spearman's Rho 缩放初始化 + AdaRound + Block/Layer Reconstruction |
| **校准方式** | 需校准数据（来自扩散采样中间状态） |
| **CUDA Kernel** | 无（纯 PyTorch STE） |
| **代码规模** | ~8 文件，约 1500 行 |

**适配可行性：低**

原因：
1. `quant_model.py` 中 `QuantModel` 的 `quant_layer_refactor_()` 硬编码了 `DiTBlock → QuantDiTBlock` 和 `FinalLayer → QuantFinalLayer` 的替换映射
2. `models.py` 完全定义了原始 DiT 架构，PTQ4DiT 的量化代码直接依赖 `models.py` 中的模块类
3. Spearman's rho 缩放计算耦合在 `models.py` 的 `Attention.qkv` / `Mlp.fc1` / `Mlp.fc2` 方法中，需要重写适配到 TinySR 的 `attn.to_q` / `ff.net[0].proj` 等
4. Block Reconstruction 的 block 结构假定（QKV 融合投影 → 与 TinySR 的分离 Q/K/V 不兼容）

**复现核心思路**：
- 在 TinySR 的 `QuantLinearW4A4` 中新增 Spearman's rho 通道缩放（替代或补充现有 smooth_scale）
- 用 AdaRound 替代 GPTQ 做逐权重自适应舍入
- 新增 Block/Layer Reconstruction 优化（需要构建逐 block 的输入输出缓存并优化 AdaRound 参数）

**预估成本：2-3 周**（高架构耦合，需大量重构）

**独特价值**：Spearman's rho（排名相关性的缩放因子计算）是与其他方法的差异化技术点。AdaRound 也可与现有 GPTQ 互补。

---

### 1.2 Q-DiT（arXiv 2406.17343）

| 维度 | 详情 |
|------|------|
| **论文** | Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers |
| **量化位宽** | W4A8（主要配置），支持 2/3/4/5/6/8/16 bit |
| **核心算法** | GPTQ（OBS 权重补偿）+ 进化搜索混合精度 group_size + 静态 EMA 激活校准 |
| **校准方式** | 需校准数据（扩散采样中间状态，256 样本） |
| **CUDA Kernel** | 无（FP32 伪量化） |
| **代码规模** | ~12 文件，约 2500 行 |

**适配可行性：低**

原因：
1. `qBlock.py` 中 `QuantDiTBlock`、`QuantAttention`、`QuantMlp` 硬编码了 DiT 的模块结构（`attn.qkv` 融合投影、`mlp.fc1/fc2`）
2. `modelutils.py` 中的 `quantize_model()` 和 `quantize_model_gptq()` 直接操作 `DiT` 模型
3. QKV 融合投影 vs 分离的 to_q/to_k/to_v 需要修改 GPTQ 的逐层处理逻辑和 Hessian 累积方式

**注意**：TinySR 现有的量化方案已经在使用 GPTQ（`models/quant/ops.py:118-222`），Q-DiT 的 GPTQ 实现与此高度相似。Q-DiT 的独特价值在于：
- **进化搜索 group_size**：为每层搜索最优 group_size ∈ [32,64,128,192,288]，用 FID 作为适应度函数
- **静态 EMA 激活校准**：预计算激活 scale，推理时不动态计算

**复现核心思路**：
- GPTQ 部分直接复用现有实现
- 将进化搜索适配到 TinySR（修改搜索空间为 TinySR 的实际层数）
- 实现 Q/K/V per-head 量化（head_dim=64）

**预估成本：2-3 周**（主要工作量在进化搜索框架的适配，GPTQ 部分可直接复用）

**独特价值**：进化搜索 group_size（数据驱动 precision allocation）、per-head Q/K/V 量化、静态 EMA 激活校准。

---

### 1.3 ViDiT-Q（ICLR 2025）

| 维度 | 详情 |
|------|------|
| **论文** | ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation |
| **量化位宽** | W8A8 / W4A8 / W4A4（支持混合精度 2/4/8 bit） |
| **核心算法** | SmoothQuant + QuaRot（Hadamard 旋转）+ CUDA GEMM Kernel |
| **校准方式** | 需少量校准数据（8 prompts 的激活统计） |
| **CUDA Kernel** | 有（W8A8/W4A8/W4A4 GEMM + Fused LayerNorm/GeLU kernels，SM80-90） |
| **代码规模** | ~25 文件，含完整 qdiff 量化框架 + CUDA 扩展 |

**适配可行性：中高**

原因：
1. `qdiff` 包设计为**可复用框架**：基于 YAML config + regex 驱动的层选择 + `QuantModel` 基类继承模式
2. 支持 PixArt-Sigma（diffusers 系），与 TinySR 同属 diffusers 生态，适配模式可参考
3. ViDiT-Q 的 `ViDiTQuantizedLinear` 独立于具体模型架构，通过 regex 匹配层名 + `apply_func_to_submodules()` 遍历
4. SmoothQuant + Hadamard 旋转的组合对 TinySR 的 SD3 架构有直接益处：
   - TinySR 当前用 smooth_scale 做平滑，ViDiT-Q 的 `channel_mask = (|W|^α) / (|A|^(1-α))` 是等效思路
   - Hadamard 旋转可以**替代 or 增强** smooth_scale，通过数学变换抹平激活峰度

**潜在问题**：
- `fast_hadamard_transform` 依赖 Linux + x86 CUDA 环境，macOS 开发需注意
- CUDA kernel（`viditq_extension`）的编译依赖 SM 架构，TinySR 的推理 GPU 需确认兼容性
- 需要处理 TinySR 中 `ff.net[0].proj` 被 GELU 类包裹的特殊结构

**复现核心思路**：
1. 配置 YAML：用 regex 匹配 TinySR 的 Attention/FFN 层名
2. 将 `ViDiTQuantizedLinear` 注入 TinySR 的 transformer_blocks
3. 收集校准数据（若干张 LR 图片的模型中间激活）
4. PTQ：计算 channel_mask + 生成 Hadamard 矩阵 + 量化权重
5. 推理：加载量化参数 + 可选 CUDA kernel 加速

**预估成本：1-2 周**（框架已模块化，主要工作在配置适配和层名 regex）

**独特价值**：SmoothQuant + Hadamard 旋转的成熟组合（已有论文验证）、CUDA 加速 kernel（实际推理速度提升）、混合精度框架、对 SD3/diffusers 系架构的 PixArt 适配先例。

---

### 1.4 DiTAS（WACV 2025）

| 维度 | 详情 |
|------|------|
| **论文** | DiTAS: Data-free Training-free Post-Training Quantization for Diffusion Transformers |
| **量化位宽** | W4A8（默认），可配 |
| **核心算法** | SVD+LoRA（交替优化 10 轮）+ TAS（时序聚合平滑）+ 逐层 alpha 网格搜索 |
| **校准方式** | **完全无数据**：仅用一次随机种子扩散采样 |
| **CUDA Kernel** | 无（纯 PyTorch） |
| **代码规模** | ~14 文件，约 1500 行 |

**适配可行性：中**

原因：
1. **TinySR 现有方案与 DiTAS 思路高度相似**：
   | 技术点 | DiTAS | TinySR 现有 (SVDQ-W4A4) |
   |--------|-------|------------------------|
   | SVD 低秩分解 | 交替 SVD-LoRA，10 轮迭代 | SVD 单次 + 可选迭代精炼 |
   | 平滑量化 | `scale = act_max^α / weight_max^(1-α)` | 完全相同公式 |
   | 残差量化 | 非对称 per-group 量化 | GPTQ 量化（对校准数据 Hessian 优化） |
   | α 搜索 | 逐层网格搜索 0.0-1.0 共 21 步 | 已支持（grid/cascade 模式） |
   | 校准数据需求 | 无（噪声驱动的扩散采样） | 需要校准图片集（8+ 张） |

2. 但也硬编码了 DiT-XL 的 28 block / 4 layers-per-block 假设
3. `LinearQuantLoRA` 的模块替换逻辑与 TinySR 的 `replace_linear_with_w4a4` 等价，但目标层名不同（`attn.qkv` vs 分离的 `to_q/to_k/to_v`）

**核心差异点分析**：
- **DiTAS 的非对称量化**（`AsymQuantizer`，值域 [0, 15]）vs TinySR 的**对称量化**（值域 [-8, 7]）——非对称对 ReLU/GELU 后激活可能更有利
- **DiTAS 的 data-free 校准**：这是一个实际优势——不需要准备校准数据集
- **TAS 时序聚合**：DiTAS 在 50 个扩散步骤间收集激活统计取 max（而非均值），TinySR 目前用固定校准数据集+单次前传

**复现核心思路**：
1. 实现 DiTAS 的 `AsymQuantizer`（非对称 per-group 量化 4-bit）
2. 将 DiTAS 的交替 SVD-LoRA（10 轮）作为 TinySR `decompose_svd_branch` 的替代/增强
3. 实现 TAS：在完整扩散采样过程中收集逐通道激活 max，聚合跨 timestep
4. 适配 DiTAS 的 `merge()` 逻辑——将 smooth_scale 吸收进 adaLN 调制权重（`shift_msa/scale_msa/shift_mlp/scale_mlp`）

**预估成本：1 周**（与现有方案高度重叠，主要是将 DiTAS 的几个增强特性嫁接进 TinySR 框架）

**独特价值**：Data-free 校准（最大差异化优势）、交替 SVD-LoRA 的收敛性、TAS 时序聚合、非对称量化（可与 TinySR 对称量化做对比实验）。

---

### 1.5 DVD-Quant（arXiv 2505.18663）

| 维度 | 详情 |
|------|------|
| **论文** | DVD-Quant: Data-free Video Diffusion Transformer Quantization via Channel Autoscaling |
| **量化位宽** | W4A4-A8（δ-GBS 动态切换），权 4-bit 固定 |
| **核心算法** | BGR（有界初始化+网格精炼）+ ARQ（Hadamard 旋转+在线自缩放）+ δ-GBS（时序自适应混合精度） |
| **校准方式** | **完全无数据**（全部在线计算或闭式求解） |
| **CUDA Kernel** | 需依赖（Block-wise scaling kernels + W4A4/W4A8 GEMM + fast Hadamard transform） |
| **代码状态** | **未开源**，仅有 README + paper |

**适配可行性：中（需从论文实现）**

原因：
1. 方法是架构无关的（论文声称），操作对象是标准 Linear 层的权重和激活
2. 但**无任何源代码**，所有实现需从论文公式推导

**逐组件分析**：

**BGR（权重无数据量化）**：
- 闭式最小二乘的 Δ, z 迭代精炼，公式清晰可实现
- 有界初始化（clamping tail outliers）是工程细节多但逻辑明确的部分
- 预期 3-4 天实现

**ARQ（激活 Hadamard 旋转+在线缩放）**：
- Hadamard 旋转库：可复用 `fast_hadamard_transform`（与 ViDiT-Q 相同依赖）
- 在线 per-channel scaling `Λ = diag(max|XH|_j)`：计算简单，需要修改 forward
- Block-wise scaling for Tensor Core：如果只做伪量化可不考虑
- 预期 3-5 天实现（伪量化版本）

**δ-GBS（时序自适应 bit-width 切换）**：
- 需要在推理循环中跟踪 `L1(F_t, F_{t-1})` 特征变化
- 累积 L1 超 δ 时从 W4A4 切到 W4A8，重置计数器
- 需要实现 W4A4 和 W4A8 两套 forward，运行时动态切换
- 预期 3-5 天实现

**预估成本：3-4 周**（从论文复现三个组件的伪量化版本，不含 CUDA kernel 优化）

**独特价值**：W4A4（极端压缩——目前五种方法中唯一声称实现 W4A4 + 视频 DiT 的）、完全 data-free、时序自适应 bit-width（创新性强）、BGR 的闭式权重精炼（不同于任何其他方法的权重量化方式）。

**最大风险**：无源码。论文公式可能缺少关键实现细节（如收敛判定、数值稳定性处理、Hadamard 矩阵尺寸对齐等），需要反复调试。

---

## 2. 综合对比矩阵

| 维度 | PTQ4DiT | Q-DiT | ViDiT-Q | DiTAS | DVD-Quant | **TinySR 现有** |
|------|---------|-------|---------|-------|-----------|-----------------|
| 适配难度 | 高 | 高 | 中 | 低 | 中 | — |
| 预估工时 | 2-3 周 | 2-3 周 | 1-2 周 | 1 周 | 3-4 周 | — |
| 量化位宽 | W4A8 | W4A8 | W4A4/A8 | W4A8 | W4A4-A8 | W4A4 |
| 校准数据 | 需要 | 需要 | 需要(少) | **不需要** | **不需要** | 需要 |
| CUDA 加速 | 无 | 无 | **有** | 无 | 需依赖 | 有(nunchaku) |
| 平滑方式 | Spearman ρ | - | Smooth+Hadamard | Smooth(L1) | Hadamard+在线 | Smooth(L1) |
| 权重量化 | AdaRound | GPTQ | 静态 per-ch | 交替SVD+AQuant | BGR(闭式) | SVD+GPTQ |
| 激活量化 | per-tensor 非对称 | per-group 非对称 | per-token 对称 | per-tensor 非对称 | per-ch 在线对称 | per-group 对称 |
| 源码可用 | ✓ | ✓ | ✓ | ✓ | **✗** | ✓ |
| 与 TinySR 方案重叠度 | 低 | GPTQ重叠 | 中 | **高** | 中 | — |

---

## 3. 推荐实施优先级

### 第 1 优先：DiTAS（快速出结果，对比已有方案）

**理由**：
- 与 TinySR 现有 SVDQ-W4A4 思路高度重叠，**1 周**可完成伪量化版本
- 核心差异化价值：**data-free 校准**（无需准备校准图集）、**TAS 时序聚合**（跨扩散步骤激活统计）、**非对称量化**（可与对称量化做对比消融）
- 可快速做对比实验，验证 data-free vs data-calibrated 差距

**实施要点**：
1. 实现 `AsymQuantizer` 替换 `UniformAffineQuantizer` 的非对称模式
2. 将交替 SVD-LoRA（10 轮）作为 `decompose_svd_branch` 的迭代参数配置
3. 实现 TAS：在扩散过程中收集跨 timestep 的逐通道 max
4. 实现 merge：将 smooth_scale 吸收进 adaLN 调制权重

### 第 2 优先：ViDiT-Q（有 CUDA 加速，工程价值高）

**理由**：
- qdiff 框架模块化程度高，**1-2 周**可完成适配
- **CUDA GEMM kernel** 可提供实际推理加速（非仅伪量化），与 nunchaku 可对比
- Hadamard 旋转是 DiTAS/DVD-Quant 都不具备的差异化技术
- PixArt 适配先例降低了 SD3 系适配的不确定性

**实施要点**：
1. 配置 TinySR 的 YAML quant config（regex 匹配层名）
2. 继承 `QuantModel` 创建 `QuantTinySR` 类
3. 适配 `ViDiTQuantizedLinear` 到 `ff.net[0].proj` 被 GELU 包裹的场景
4. 编译 CUDA kernel 并验证兼容性

### 第 3 优先：DVD-Quant（创新性强，但成本高）

**理由**：
- W4A4 是五种方法中的极限压缩水平，与 TinySR 的 W4A4 目标一致
- δ-GBS 时序自适应是独特创新，可能显著改善视频/高步数场景
- 但**无源码**，需从论文实现 **3-4 周**
- 建议等 DiTAS + ViDiT-Q 完成后，若有进一步需要再投入

### 第 4/5 优先：PTQ4DiT / Q-DiT（技术价值弱于前者）

- Q-DiT 的 GPTQ 部分 TinySR 已具备，进化搜索成本高昂
- PTQ4DiT 的 Spearman's rho 可能是唯一值得单独提取的技术点（可嵌入 DiTAS 的平滑模块中尝试）

---

## 4. 建议的对比实验方案（伪量化对比基线）

在实现各方法后，统一在以下维度对比：

| 对比维度 | 具体指标 |
|----------|----------|
| 图像质量 | PSNR / SSIM / LPIPS / MANIQA（相比 FP16 baseline） |
| 量化误差 | 逐层 weight MSE、逐层 output MSE vs FP16 |
| 层元数据 | 逐层 smooth_alpha、SVD 误差、量化 scale 分布 |
| 推理速度 | 平均推理时间（纯 PyTorch forward，不含 VAE） |
| 激活分析 | 各层 SNR / 峰度 / 异常值比率（用现有 ActivationErrorAnalyzer） |
| 校准依赖 | 是否需要校准数据、校准数据量和采样策略 |

---

## 5. 需要解答的问题（当前不确定因素）

1. **GPU 环境**：ViDiT-Q 的 CUDA kernel 需要 SM80+ GPU（A100/3090/4090），当前环境是否满足？
2. **Hadamard transform 依赖**：`fast_hadamard_transform` 只有 Linux + x86 CUDA 的预编译包，macOS 上开发需如何处理（用纯 Python 回退？）？
3. **非对称 vs 对称量化**：TinySR 现有方案是 symmetric (-8,7)，但 PTQ4DiT、DiTAS、Q-DiT 默认都是 asymmetric。对比实验时是否需要统一量化范围？
4. **W4A4 vs W4A8**：部分方法默认 W4A8（DiTAS/PTQ4DiT/Q-DiT），部分支持 W4A4（ViDiT-Q/DVD-Quant）。是否所有方法统一做 W4A8 对比，还是按其最佳配置各显神通？
5. **Nunchaku 对齐**：伪量化对比验证后，是否需要将优胜方法的权重也导出为 nunchaku 格式做实际加速对比？
