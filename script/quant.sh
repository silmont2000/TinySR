#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
#  W4A4 quantized inference + calibration for TinySR
#  Entry point: train/train_quant.py
# =============================================================================

python train/train_quant.py  \
  --rank 64                    `# LoRA rank for transformer adapter` \
  --w_bits 4                   `# weight quantization bits (default: 4)` \
  --a_bits 8                   `# activation quantization bits (default: 4)` \
  --svdq_rank 32               `# SVD low-rank branch rank (default: 32)` \
  --quant_scope dit_full       `# none | ffn_only | attn_only | dit_full` \
  --calib_input_dir /data/disk2/xby/TinySR/dataset/DIV2K_train_patches/pick \
  --calib_images 100           `# number of calibration images (default: 8)` \
  --input_dir /data/disk2/xby/TinySR/dataset/test_image \
  --svdq_smooth_alpha 0.55 \
  --save_quant_meta            `# save full quant_meta in w4a4_report.json` \
  --save_quant_state outputs/quant_state.pt
  # --search_smooth_alpha cascade --svdq_alpha_grid 22 \
  # --save_model_path            `# export torchao model (auto path in output_dir)`

# =============================================================================
#  smooth_alpha 三级优先方案
# =============================================================================
#
#  最高优 --load_smooth_alpha_report <json>
#    从之前运行的 w4a4_report.json 加载逐层 alpha。
#    忽略 --svdq_smooth_alpha 和 --search_smooth_alpha。
#    不计算误差，不用级联。校准完直接推理。
#
#  次高优 --svdq_smooth_alpha <float>  (default: 0.5)
#    所有层统一使用此 alpha。
#    不计算误差，不用级联。校准完直接推理。
#    （如果同时设了 --search_smooth_alpha，则按最低优逻辑搜索）
#
#  最低优 --search_smooth_alpha [{grid|cascade}]
#    搜索每层最优 alpha。
#    grid     单轮网格搜索（默认，等效于 --search_smooth_alpha）
#    cascade  逐层级联冻结（需额外设 --cascade_calib_images）
#    搜索时自动计算重构误差。
#
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
# --svdq_iterations <int>           SVD迭代细化次数 (default: 0)
# --quant_scope <str>               量化范围: none | ffn_only | attn_only | dit_full
# --quant_config <json>             逐层量化配置文件 (可选)
# --calib_images <int>              校准图像总数 (default: 8)
#
# ---- smooth_alpha 控制 (三级优先) ----
# --load_smooth_alpha_report <json> 最高优：从报告加载逐层alpha (覆盖下述两个)
# --svdq_smooth_alpha <float>       次高优：所有层统一alpha (default: 0.5)
# --search_smooth_alpha [{grid|cascade}]  最低优：搜索最优alpha
# --svdq_alpha_grid <int>           alpha搜索网格点数 (default: 11)
# --cascade_calib_images <int>      cascade模式的每层级联校准图像数 (default: 4)
#
# ---- LoRA ----
# --rank <int>                      LoRA rank (default: 64)
#
# ---- 校准缓存 ----
# --calib_cache <pt_path>           校准缓存路径 (存在则跳过校准直接加载)
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
# --save_quant_meta                 在w4a4_report.json中保存完整quant_meta
#
# ---- 激活误差分析 ----
# --analyze_activation              启用逐层激活量化误差分析
# --analyze_max_samples <int>       每层最大样本数 (default: 200)
# --analyze_max_points <int>        每样本最大点数 (default: 20000)
#
# =============================================================================
#  常用变体 (取消注释使用)
# =============================================================================
#
# --- 固定统一alpha，不搜索 (最快) ---
# python train/train_quant.py \
#   --quant_scope attn_only \
#   --svdq_smooth_alpha 0.55 \
#   --calib_input_dir dataset/StableSR_testsets/DrealSRVal_crop128/test_LR \
#   --calib_images 8
#
# --- 加载历史报告 (最高优) ---
# python train/train_quant.py \
#   --quant_scope dit_full \
#   --load_smooth_alpha_report outputs/w4a4_report.json \
#   --calib_input_dir dataset/StableSR_testsets/DrealSRVal_crop128/test_LR
#
# --- 网格搜索最优alpha ---
# python train/train_quant.py \
#   --quant_scope attn_only \
#   --search_smooth_alpha grid \
#   --svdq_alpha_grid 11 \
#   --calib_images 16
#
# --- 逐层级联搜索 ---
# python train/train_quant.py \
#   --rank 64 --w_bits 4 --a_bits 8 \
#   --quant_scope dit_full \
#   --search_smooth_alpha cascade \
#   --cascade_calib_images 4 \
#   --calib_images 16
#
# --- 加载预导出模型 (跳过校准) ---
# python train/train_quant.py \
#   --quant_scope dit_full \
#   --input_dir dataset/test_image/ \
#   --load_model_path outputs/torchao_model.pt
#
# --- 保存完整量化state_dict + 下次直接加载 (推荐) ---
# # 第一次：校准并保存
# python train/train_quant.py \
#   --quant_scope dit_full \
#   --search_smooth_alpha cascade \
#   --calib_input_dir dataset/StableSR_testsets/DrealSRVal_crop128/test_LR \
#   --save_quant_state outputs/quant_state.pt
# # 第二次：加载后直接推理，无需校准
# python train/train_quant.py \
#   --quant_scope dit_full \
#   --input_dir dataset/test_image/ \
#   --load_quant_state outputs/quant_state.pt
#
# --- 激活误差分析 ---
# python train/train_quant.py \
#   --rank 64 --w_bits 4 --a_bits 8 \
#   --quant_scope dit_full \
#   --analyze_activation \
#   --analyze_max_samples 100
