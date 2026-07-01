#!/bin/bash
set -e

# ============================================================
# Sweep: attn_only + selected FFN blocks → find quant boundary
# ============================================================
#
# Configs are named "label|ffn_blocks_comma_string"
#   attn_only     — baseline, no FFN layers quantized
#   tier_a        — best FFN blocks: 0,2,3,4,5,7   (SVD < 0.003)
#   tier_a_b      — tier_a + medium: add 1,6,8     (SVD < 0.004)
#   dit_full      — all 12 FFN blocks              (upper bound)

CONFIGS=(
    "attn_only|"
    "tier_a0|0"
    "tier_a1|1"
    "tier_a2|2"
    "tier_a12|1,2"
    "tier_a01|0,1"
    "tier_a02|0,2"
    "tier_a3|3"
    "tier_a4|4"
    "tier_a5|5"
    "tier_a34|3,4"
    "tier_a35|3,5"
    "tier_a45|4,5"
    "tier_a6|6"
    "tier_a7|7"
    "tier_a8|8"
    "tier_a67|6,7"
    "tier_a68|6,8"
    "tier_a78|7,8"
    "tier_a9|9"
    "tier_a10|10"
    "tier_a11|11"
    "tier_a910|9,10"
    "tier_a911|9,11"
    "tier_a1011|10,11"
)

RESULTS_FILE="logs/sweep_ffn_blocks_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p logs

echo "Sweep started at $(date)"
echo "Results: $RESULTS_FILE"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    NAME="${CONFIG%%|*}"
    BLOCKS="${CONFIG##*|}"

    echo "======================================================"
    echo "  Running: $NAME  (ffn_blocks=${BLOCKS:-none})"
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
        --svdq_rank 32 \
        --svdq_smooth_alpha 1 --cascade_calib_images 16 \
        --rank 64 --save_quant_meta \
        --quant_ffn_blocks "$BLOCKS" \
        --output_dir "$TRAIN_OUT"

    # ── Step 2: nunchaku inference ────────────────────────────
    python test/test_nunchaku_inference.py \
        --pretrained_model_name_or_path "$TRAIN_OUT/merged_backbone" \
        --nunchaku_state "$TRAIN_OUT/nunchaku.safetensors" \
        --quant_scope attn_only --rank 32 \
        --input_dir ../RealSR/LR \
        --quant_ffn_blocks "$BLOCKS" \
        --output_dir "$INF_OUT"

    # test_nunchaku_inference appends a timestamp to output_dir
    INF_ACTUAL=$(ls -dt "${INF_OUT}"*/ 2>/dev/null | head -1)
    if [ -z "$INF_ACTUAL" ]; then
        echo "  [ERROR] inference output dir not found under ${INF_OUT}*"
        exit 1
    fi
    echo "  inference output: $INF_ACTUAL"

    # ── Step 3: metrics ───────────────────────────────────────
    conda activate tinysr_quant

    {
        echo "===== Config: $NAME (ffn_blocks=${BLOCKS:-none}) ====="
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
