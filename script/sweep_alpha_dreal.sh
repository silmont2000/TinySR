#!/bin/bash
# ==============================================================================
# Sweep svdq_smooth_alpha for W4A8 quantized TinySR
# Usage:  cd /data/disk2/xby/TinySR && bash script/sweep_alpha_dreal.sh
# ==============================================================================

set -euo pipefail
cd /data/disk2/xby/TinySR

# ========== Config ==========
INPUT_DIR="dataset/StableSR_testsets/DrealSRVal_crop128/test_LR"
GT_DIR="dataset/StableSR_testsets/DrealSRVal_crop128/test_HR"
CALIB_DIR="dataset/StableSR_testsets/DrealSRVal_crop128/test_LR"

W_BITS=4
A_BITS=8
SVDQ_RANK=32
QUANT_SCOPE="dit_full"
CALIB_IMAGES=100
DEVICE="cuda"

MIXED_PRECISION="fp16"
RANK=64

ALPHAS="0.55"

RESULTS_FILE="logs/sweep_alpha_results_dreal.txt"
METRICS_LOG_DIR="logs/metrics/dreal"
mkdir -p "${METRICS_LOG_DIR}"

# ========== Run inference for each alpha ==========
echo "==================== Step 1: Inference ===================="
OUTPUT_DIRS=""
FAILED=""

for alpha in ${ALPHAS}; do
    alpha_int=$(python3 -c "print(int(${alpha} * 100))")
    dir="outputs/dreal/w${W_BITS}a${A_BITS}_svdq_r${SVDQ_RANK}_${QUANT_SCOPE}_a${alpha_int}"

    echo ">>> alpha=${alpha} -> ${dir}"
    if python train/infer_w4a4_tinysr.py \
        --rank "${RANK}" \
        --w_bits "${W_BITS}" \
        --a_bits "${A_BITS}" \
        --svdq_rank "${SVDQ_RANK}" \
        --svdq_smooth_alpha "${alpha}" \
        --quant_scope "${QUANT_SCOPE}" \
        --calib_images "${CALIB_IMAGES}" \
        --input_dir "${INPUT_DIR}" \
        --calib_input_dir "${CALIB_DIR}" \
        --output_dir "${dir}" \
        --device "${DEVICE}" \
        --mixed_precision "${MIXED_PRECISION}"; then
        OUTPUT_DIRS="${OUTPUT_DIRS} ${dir}"
        echo "  [OK]"
    else
        echo "  [FAIL]"
        FAILED="${FAILED} ${alpha}"
    fi
    echo ""
done

if [ -z "${OUTPUT_DIRS}" ]; then
    echo "ERROR: no successful runs"
    exit 1
fi

# ========== Run metrics on all outputs ==========
echo "==================== Step 2: Metrics ===================="

INP=()
GT=()
for d in ${OUTPUT_DIRS}; do
    INP+=("${d}")
    GT+=("${GT_DIR}")
done

TIMESTAMP=$(date +%y%m%d-%H%M%S)
METRICS_LOG="${METRICS_LOG_DIR}/sweep_${TIMESTAMP}.log"

python test/test_metrics.py \
    --inp_imgs "${INP[@]}" \
    --gt_imgs "${GT[@]}" \
    --log "${METRICS_LOG_DIR}" 2>&1 | tee "${METRICS_LOG}"

# ========== Collect results ==========
echo "==================== Step 3: Collect ===================="

{
    echo "# Alpha Sweep Results - $(date)"
    echo "# Config: w${W_BITS}a${A_BITS} svdq_r${SVDQ_RANK} scope=${QUANT_SCOPE}"
    echo "# Dataset: ${INPUT_DIR}"
    echo "# Alphas: ${ALPHAS}"
    if [ -n "${FAILED}" ]; then
        echo "# Failed: ${FAILED}"
    fi
    echo ""

    # Extract the Average Metrics blocks from the metrics output
    awk '/^===== Average Metrics for/{flag=1} flag{print; if(/FID Runtime:/){print ""; flag=0}}' "${METRICS_LOG}"
} > "${RESULTS_FILE}"

echo ""
echo "==================== Done ===================="
echo "Summary: ${RESULTS_FILE}"
cat "${RESULTS_FILE}"
