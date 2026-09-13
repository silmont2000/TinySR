#!/bin/bash
# =============================================================================
# Supplement runs for compare.tex: Min-Max / SmoothQuant / ViDiT-Q / SVD-full / Ours
# at W8A8, W4A8, W8A4. Outputs go to system disk (/root/quant_extra).
#
#  - Min-Max / SmoothQuant strictly follow the paper: --no_gptq
#  - Ours = SVDQ-Global alpha=0.8, attn_only + FFN blocks {1,6,9,10,11}
#  - --save_merged_backbone "" and --save_nunchaku "" keep the full data disk safe
#
# Usage: bash run_quant_supplement.sh [smoke_lr_dir]   (arg = optional input subset)
#        RERUN_W4A4=1 bash run_quant_supplement.sh    (also rerun W4A4 for consistency)
# =============================================================================
set -uo pipefail

INPUT_DIR=${1:-/root/autodl-tmp/RealSR/LR}   # optional small subset dir for smoke test
set --                                      # clear positional args (conda activate chokes on $1)

cd /root/autodl-tmp/TinySR
source /root/miniconda3/bin/activate
conda activate tinysr_nunchaku
export PYTHONUNBUFFERED=1

LR=/root/autodl-tmp/RealSR/LR
GT=/root/autodl-tmp/RealSR/HR
OUT_ROOT=/root/quant_extra
LOG_DIR=${OUT_ROOT}/metrics_logs
RUNLOG=${OUT_ROOT}/batch_run.log
mkdir -p "${LOG_DIR}"

RERUN_W4A4=${RERUN_W4A4:-0}
ONLY_W4A4=${ONLY_W4A4:-0}     # 1 = skip the 15-run main batch, only run the W4A4 consistency runs
LIMIT=${LIMIT:-0}              # 0 = run all; N = run at most N experiments
n=0

COMMON=(
  --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr
  --vae_path checkpoint/vae/separable
  --lora_dir checkpoint/tinysr
  --embedding_dir dataset/default/
  --cache_dir /root/hf_cache
  --mixed_precision fp16
  --upscale 4
  --process_size 512
  --align_method adain
  --warmup_images 1
  --device cuda
  --save_merged_backbone ""
  --save_nunchaku ""
  --input_dir "${INPUT_DIR}"
  --calib_input_dir "${LR}"
)

run_one() {
  local name=$1 w=$2 a=$3
  shift 3
  local out="${OUT_ROOT}/${name}_w${w}a${a}"
  echo "[$(date '+%F %T')] START ${name} w${w}a${a} -> ${out}" | tee -a "${RUNLOG}"
  python train/train_quant.py "${COMMON[@]}" \
      --w_bits "${w}" --a_bits "${a}" --output_dir "${out}" "$@" \
      >> "${RUNLOG}" 2>&1
  local rc1=$?
  if [ ${rc1} -eq 0 ]; then
    python test/test_metrics.py --inp_imgs "${out}" --gt_imgs "${GT}" --log "${LOG_DIR}" \
        >> "${RUNLOG}" 2>&1
    local rc2=$?
  else
    local rc2=99
  fi
  echo "[$(date '+%F %T')] END ${name} w${w}a${a} rc_infer=${rc1} rc_metrics=${rc2}" | tee -a "${RUNLOG}"
}

maybe_run() {
  n=$((n + 1))
  if [ "${LIMIT}" -gt 0 ] && [ "${n}" -gt "${LIMIT}" ]; then
    return 0
  fi
  run_one "$@"
}

SCOPE=(--quant_scope attn_only --quant_ffn_blocks 1,6,9,10,11)

echo "[$(date '+%F %T')] BATCH START input=${INPUT_DIR} rerun_w4a4=${RERUN_W4A4}" | tee -a "${RUNLOG}"

if [ "${ONLY_W4A4}" != "1" ]; then
  for prec in "8 8" "4 8" "8 4"; do
    set -- ${prec}
    maybe_run minmax "$1" "$2" --svdq_rank 0 --no_smooth --no_gptq "${SCOPE[@]}" --calib_images 100
    maybe_run smoothquant "$1" "$2" --svdq_rank 0 --no_gptq --svdq_smooth_alpha 0.5 "${SCOPE[@]}" --calib_images 100
    maybe_run viditq "$1" "$2" --viditq --viditq_alpha 0.5 "${SCOPE[@]}" --calib_images 10
    maybe_run svd_full "$1" "$2" --svdq_rank 32 --svdq_smooth_alpha 0.8 --quant_scope dit_full --weight_group_size 64 --calib_images 100
    maybe_run ours "$1" "$2" --svdq_rank 32 --svdq_smooth_alpha 0.8 "${SCOPE[@]}" --weight_group_size 64 --calib_images 100
  done
fi

if [ "${RERUN_W4A4}" = "1" ] || [ "${ONLY_W4A4}" = "1" ]; then
  echo "[$(date '+%F %T')] RERUN W4A4 (no_gptq consistency)" | tee -a "${RUNLOG}"
  maybe_run minmax 4 4 --svdq_rank 0 --no_smooth --no_gptq "${SCOPE[@]}" --calib_images 100
  maybe_run smoothquant 4 4 --svdq_rank 0 --no_gptq --svdq_smooth_alpha 0.5 "${SCOPE[@]}" --calib_images 100
  maybe_run viditq 4 4 --viditq --viditq_alpha 0.5 "${SCOPE[@]}" --calib_images 10
fi

echo "[$(date '+%F %T')] BATCH DONE" | tee -a "${RUNLOG}"
