#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
#  W4A4 quantized inference + calibration for TinySR
#  Entry point: train/train_quant.py
# =============================================================================

python train/train_quant.py  \
  --w_bits 4                   `# weight quantization bits (default: 4)` \
  --a_bits 8                   `# activation quantization bits (default: 4)` \
  --svdq_rank 32               `# SVD low-rank branch rank (default: 32)` \
  --quant_scope dit_full       `# none | ffn_only | attn_only | dit_full` \
  --calib_input_dir /data/disk2/xby/TinySR/dataset/DIV2K_train_patches/pick \
  --calib_images 100           `# number of calibration images (default: 8)` \
  --input_dir /data/disk2/xby/TinySR/dataset/StableSR_testsets/DIV2K_V2_val/test_LR \
  --search_smooth_alpha        `# per-layer alpha grid search` \
  --layer_cascade_smooth_alpha `# layer-by-layer cascade freeze` \
  --save_quant_meta            `# save full quant_meta in w4a4_report.json` \
  # --save_model_path            `# export torchao model (auto path in output_dir)`

# =============================================================================
#  全部可用选项
# =============================================================================
#
# ---- 模型路径 ----
# --pretrained_model_name_or_path <path>    默认: checkpoint/tinybackbone/prune-12-merge-tinysr
# --vae_path <path>                         默认: checkpoint/vae/separable
# --lora_dir <path>                         默认: checkpoint/tinysr
# --cache_dir <path>                        默认: /data/disk2/xby/models
#
# ---- 输入/输出 ----
# --input_dir <path>                推理输入图像目录 (default: dataset/test_image/)
# --calib_input_dir <path>          校准图像目录
# --output_dir <path>               输出目录 (默认自动生成)
# --embedding_dir <path>            prompt embedding 目录 (default: dataset/default/)
#
# ---- 量化参数 ----
# --w_bits <int>                    权重量化位宽 (default: 4)
# --a_bits <int>                    激活量化位宽 (default: 4)
# --svdq_rank <int>                 SVD低位分支秩 (default: 32)
# --svdq_smooth_alpha <float>       smooth量化迁移强度 (default: 0.5)
# --svdq_no_error                   跳过smooth_alpha错误日志 (default: False)
# --search_smooth_alpha             逐层alpha网格搜索 (default: False)
# --layer_cascade_smooth_alpha     逐层级联冻结 (default: False)
# --svdq_alpha_grid <int>           alpha网格搜索点数，越小越快 (default: 7，旧默认11)
# --cascade_calib_images <int>      级联每层校准图像数 (default: 4)
# --quant_scope <str>               量化范围: none | ffn_only | attn_only | dit_full
# --quant_config <json>             逐层量化配置文件 (可选)
# --calib_images <int>              校准图像总数 (default: 8)
#
# ---- LoRA ----
# --rank <int>                      LoRA rank (default: 64)
#
# ---- 校准缓存 ----
# --calib_cache <pt_path>           校准缓存路径 (存在则跳过校准直接加载)
# --load_smooth_alpha_report <json> 从之前运行的w4a4_report.json加载逐层alpha (覆盖搜索)
#
# ---- 模型导出/预加载 ----
# --save_model_path <path>          导出torchao量化模型 (用=__auto__自动路径)
# --load_model_path <pt_path>       加载已导出的torchao模型 (跳过校准+LoRA)
#
# ---- 推理 ----
# --device <str>                    设备 (default: cuda)
# --mixed_precision <str>           精度: fp16 | fp32 (default: fp16)
# --upscale <int>                   放大倍数 (default: 4)
# --process_size <int>              处理尺寸 (default: 512)
# --align_method <str>              颜色对齐: wavelet | adain | nofix (default: adain)
# --warmup_images <int>             预热图像数 (default: 1)
# --latent_tiled_size <int>         latent分块大小 (default: 64)
# --latent_tiled_overlap <int>      latent分块重叠 (default: 8)
# --timestep <float>                扩散步数 (default: 1000.0)
#
# ---- 时序报告 ----
# --save_quant_meta                 在w4a4_report.json中保存完整quant_meta (default: False)
#
# ---- 激活误差分析 ----
# --analyze_activation              启用逐层激活量化误差分析 (default: False)
# --analyze_max_samples <int>       每层最大样本数 (default: 200)
# --analyze_max_points <int>        每样本最大点数 (default: 20000)
#
# =============================================================================
#  常用变体 (取消注释使用)
# =============================================================================
#
# --- 快速测试 (轻量校准) ---
# python train/train_quant.py \
#   --quant_scope attn_only \
#   --calib_input_dir dataset/StableSR_testsets/DrealSRVal_crop128/test_LR \
#   --calib_images 4 \
#   --svdq_no_error
#
# --- 加载预导出模型 (跳过校准) ---
# python train/train_quant.py \
#   --quant_scope dit_full \
#   --input_dir dataset/test_image/ \
#   --load_model_path outputs/torchao_model.pt
#
# --- 逐层配置文件 + 平滑alpha报告 ---
# python train/train_quant.py \
#   --quant_config train/layer_quant_config_example.json \
#   --load_smooth_alpha_report outputs/w4a4_report.json \
#   --calib_input_dir dataset/StableSR_testsets/DrealSRVal_crop128/test_LR \
#   --calib_cache outputs/calib_cache.pt
#
# --- 激活误差分析 ---
# python train/train_quant.py \
#   --rank 64 --w_bits 4 --a_bits 8 \
#   --quant_scope dit_full \
#   --analyze_activation \
#   --analyze_max_samples 100
