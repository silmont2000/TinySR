python train/infer_w4a4_tinysr.py  \
--rank 64  \
--svdq_rank 32  \
--svdq_smooth_alpha 1.0  \
--quant_scope ffn_only  \
--calib_images 100  \
--input_dir dataset/test_image/  \
--output_dir outputs/w4a4_svdq_fixed_r32_ffn_a100  \
--save_quant_meta