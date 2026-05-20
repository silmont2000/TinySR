#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
#  W4A4 quantized MACs for TinySR
#  Entry point: test/test_quant_mac.py
#
#  Usage:
#    bash script/quant_mac.sh [extra_python_args...]
#
#  Compare with unquantized baseline:
#    python test/test_mac.py \
#      --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr \
#      --vae_path checkpoint/vae/separable \
#      --lora_dir checkpoint/tinysr
#
#  Both scripts use the same dummy input (1,3,128,128) -> (512,512)
#  and same VAE+transformer backbone.
#  The numbers are directly comparable: quantized MACs ≈ unquantized MACs
#  + optional LowRankBranch overhead.
# =============================================================================

python test/test_quant_mac.py \
  --load_quant_state outputs/quant_state.pt \
  --quant_scope dit_full \
  --w_bits 4 \
  --a_bits 8 \
  --svdq_rank 32 \
  --svdq_smooth_alpha 0.55 \
  --rank 64

echo ""
echo "=== Compare with unquantized baseline ==="
echo "  python test/test_mac.py \\"
echo "    --pretrained_model_name_or_path checkpoint/tinybackbone/prune-12-merge-tinysr \\"
echo "    --vae_path checkpoint/vae/separable \\"
echo "    --lora_dir checkpoint/tinysr"
