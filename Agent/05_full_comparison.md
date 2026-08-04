# TinySR W4A4 量化方法全景对比

\begin{table*}[!t]
\setlength{\abovecaptionskip}{0.1cm}
\caption{%
Quantitative comparison of W4A4 quantization methods on RealSR.
``w/o quant.'' denotes the full-precision (FP16) baseline.
\textbf{Bold} indicates the best result among all quantized methods (excluding FP16).
}
\label{tab:w4a4_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l|ccccccccc@{}}
\toprule
\textbf{Method} & \textbf{PSNR} $\uparrow$ & \textbf{SSIM} $\uparrow$ & \textbf{LPIPS} $\downarrow$ & \textbf{DISTS} $\downarrow$ & \textbf{CLIPIQA} $\uparrow$ & \textbf{NIQE} $\downarrow$ & \textbf{MUSIQ} $\uparrow$ & \textbf{MANIQA} $\uparrow$ & \textbf{FID} $\downarrow$ \\
\midrule
w/o quant.                & 24.79 & 0.7171 & 0.2806 & 0.2123 & 0.7035 & 4.74 & 69.78 & 0.6235 & 118.08 \\
\midrule
SVDQ-mix                  & \textbf{24.89} & \textbf{0.7116} & \textbf{0.2839} & \textbf{0.2145} & 0.7070 & 4.66 & \textbf{69.08} & \textbf{0.6060} & \textbf{114.07} \\\cmidrule{2-10}
SVDQ-full ($\alpha{=}0.8$) & 24.87 & 0.7016 & 0.2961 & 0.2258 & 0.7000 & 4.64 & 68.31 & 0.5786 & 122.80 \\
SmoothQuant                & 24.62 & 0.6937 & 0.3008 & 0.2296 & 0.7066 & 4.68 & 68.57 & 0.5870 & 121.81 \\
Min-Max                    & 24.60 & 0.6785 & 0.3254 & 0.2499 & 0.6882 & \textbf{4.63} & 66.54 & 0.5578 & 140.43 \\
ViDiT-Q                    & 24.71 & 0.6934 & 0.3088 & 0.2319 & \textbf{0.7237} & 4.82 & 68.50 & 0.5821 & 125.35 \\
DiTAS W4A4                 & 23.81 & 0.6104 & 0.4722 & 0.3363 & 0.6513 & \textbf{4.63} & 56.01 & 0.4024 & 240.25 \\
\bottomrule
\end{tabular}%
}
\end{table*}

## 关键发现

1. **SVDQ-Global α=0.8 全面最优**：9 项指标中 5 项第一，FID 甚至超越 FP16。当前管线的最强配置
2. **三种 α 搜索策略差异小**：Global/Independent/Progressive 的 FID 仅差 2.4 点，Global 简单且有效
3. **MinMax 明显最差**：FID 140（+26 vs SVDQ），无平滑量化不可用
4. **per-tensor W4A4 不可行**：DiTAS W4A4 的 FID 240、PSNR -1.0，per-tensor 16 个离散值完全无法覆盖激活分布
5. **ViDiT-Q W4A4 中等**：FID 125.3（-11 vs SVDQ），CLIPIQA 反而最高。Hadamard 旋转有效但有精度代价
6. **SVDQ-dit_full vs others**：全层量化（83 层）vs attn+5FFN（46 层），质量仅差 ~1 FID，说明 SVD+GPTQ 补偿足够

## 配置对照

|                    | Global α | Indep.    | Prog.     | dit_full           | Smooth    | MinMax    | ViDiT-Q   | DiTAS W4A4 |
| ------------------ | --------- | --------- | --------- | ------------------ | --------- | --------- | --------- | ---------- |
| **SVD 分支** | r=32      | r=32      | r=32      | r=32               | 无        | 无        | 无        | r=32×10   |
| **GPTQ**     | ✓        | ✓        | ✓        | ✓                 | ✗        | ✗        | ✗        | ✗         |
| **平滑**     | α=0.8    | grid      | cascade   | α=0.8             | α=0.5    | 无        | S+H       | grid 21    |
| **激活量化** | g64       | g64       | g64       | g64                | g64       | g64       | token     | tensor     |
| **对称**     | sym       | sym       | sym       | sym                | sym       | sym       | w非/a对   | 非对称     |
| **范围**     | attn+5FFN | attn+5FFN | attn+5FFN | **dit_full** | attn+5FFN | attn+5FFN | attn+5FFN | attn+5FFN  |
| **校准**     | 真实图    | 真实图    | 真实图    | 真实图             | 真实图    | 真实图    | 真实8张   | 随机50     |
