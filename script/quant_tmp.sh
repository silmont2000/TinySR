python train/infer_w4a4_tinysr.py  \
--rank 64  \
--w_bits 4  \
--a_bits 8 \
--svdq_rank 32  \
--quant_scope dit_full  \
--calib_input_dir /data/disk2/xby/TinySR/dataset/DIV2K_train_patches/pick \
--calib_images 100  \
--input_dir /data/disk2/xby/TinySR/dataset/test_image  \
--layer_cascade_smooth_alpha \
--load_smooth_alpha_report /data/disk2/xby/TinySR/outputs/w4a8_svdq_r32_dit_full_alayer_cascade_real/w4a4_report.json  \
--svdq_no_error \
--save_quant_meta 
# --search_smooth_alpha  \
# --svdq_smooth_alpha 0.55  \
# --save_model_path \
# --load_model_path outputs/tinysr_w4a8_quantized.pt  # skip calibration, load saved model
# --search_smooth_alpha  \
# --quant_config /data/disk2/xby/TinySR/train/layer_quant_config_example.json  \
# --calib_input_dir dataset/StableSR_testsets/DrealSRVal_crop128/test_LR \
# --calib_cache outputs/calib_cache.pt  \
