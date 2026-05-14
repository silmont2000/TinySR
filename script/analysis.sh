# cd /data/disk2/xby/TinySR
# python train/pre_analysis.py \
#   --output_dir outputs/act_smoothing_calib_debug \
#   --max_images_per_set 0 \
#   --max_qq_points 8000 \
#   --max_ks_pairs_per_layer 12 \
#   --mixed_precision fp16 \
#   --device cuda
#   激活值分析

python train/infer_w4a4_tinysr.py  \
--rank 64  \
--w_bits 4  \
--a_bits 4 \
--svdq_rank 32  \
--svdq_smooth_alpha 0.98  \
--calib_images 100  \
--calib_cache outputs/calib_cache_w4a4_smoke.pt  \
--input_dir dataset/test_image/ \
--calib_input_dir dataset/test_image/  \
--quant_scope ffn_only \
--analyze_activation \
--analyze_max_samples 200 \
--analyze_max_points 20000000