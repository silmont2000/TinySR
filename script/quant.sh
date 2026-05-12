python train/infer_w4a4_tinysr.py  \
--rank 64  \
--w_bits 4  \
--a_bits 4 \
--svdq_rank 32  \
--svdq_smooth_alpha 0.99  \
--quant_scope ffn_only  \
--calib_images 100  \
--input_dir dataset/test_image/  \
--save_quant_meta