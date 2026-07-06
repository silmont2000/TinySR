#!/bin/bash
set -e

# ============================================================
# Sweep: attn_only + blocks {6,8,9,10,11}, varying smooth_alpha
# ============================================================

FFN_BLOCKS="6,8,9,10,11"

ALPHAS=(
    "0.5"
    "0.6"
    "0.7"
    "0.8"
    "0.9"
    "1.0"
)

RESULTS_FILE="logs/sweep_alpha_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p logs

echo "Sweep started at $(date)"
echo "Results: $RESULTS_FILE"
echo ""

for ALPHA in "${ALPHAS[@]}"; do

    NAME="alpha_${ALPHA}"

    echo "======================================================"
    echo "  Running: $NAME"
    echo "======================================================"

    TRAIN_OUT="outputs/sweep_${NAME}"
    INF_OUT="outputs/sweep_nunchaku_${NAME}"

    # ── Step 1: train + quant ─────────────────────────────────
    cd /root/autodl-tmp/TinySR
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate tinysr_nunchaku

    python train/train_quant.py \
        --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr \
        --vae_path checkpoint/vae/separable \
        --lora_dir checkpoint/tinysr \
        --calib_input_dir ../RealSR/LR \
        --calib_images 100 \
        --w_bits 4 --a_bits 4 --align_nunchaku_inference \
        --quant_scope attn_only \
        --quant_ffn_blocks "$FFN_BLOCKS" \
        --svdq_smooth_alpha "$ALPHA" \
        --rank 64 --save_quant_meta \
        --output_dir "$TRAIN_OUT"

    # ── Step 2: nunchaku inference ────────────────────────────
    python test/test_nunchaku_inference.py \
        --pretrained_model_name_or_path "$TRAIN_OUT/merged_backbone" \
        --nunchaku_state "$TRAIN_OUT/nunchaku.safetensors" \
        --quant_scope attn_only --rank 32 \
        --input_dir ../RealSR/LR \
        --quant_ffn_blocks "$FFN_BLOCKS" \
        --output_dir "$INF_OUT"

    INF_ACTUAL=$(ls -dt "${INF_OUT}"*/ 2>/dev/null | head -1)
    if [ -z "$INF_ACTUAL" ]; then
        echo "  [ERROR] inference output dir not found under ${INF_OUT}*"
        exit 1
    fi
    echo "  inference output: $INF_ACTUAL"

    # ── Step 3: metrics ───────────────────────────────────────
    conda activate tinysr_quant

    {
        echo "===== Config: $NAME (alpha=$ALPHA, ffn_blocks=$FFN_BLOCKS) ====="
        HF_HUB_OFFLINE=1 python test/test_metrics.py \
            --inp_imgs "$INF_ACTUAL" \
            --gt_imgs ../RealSR/HR \
            --log logs/metrics 2>&1 | \
        awk '/===== Average Metrics/,/===== Evaluation Completed/'
        echo ""
    } | tee -a "$RESULTS_FILE"

    echo ""
    echo "  Done: $NAME"
    echo ""
done

echo "======================================================"
echo "  Sweep complete."
echo "  Results: $RESULTS_FILE"
echo "======================================================"
