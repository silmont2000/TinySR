# === Full pipeline: calibrate + save model ===
python train/infer_w4a4_tinysr.py  \
--rank 64  \
--w_bits 4  \
--a_bits 8 \
--svdq_rank 32  \
--quant_scope dit_full  \
--calib_input_dir /data/disk2/xby/TinySR/dataset/test_image \
--calib_images 100  \
--input_dir /data/disk2/xby/TinySR/dataset/test_image  \
--search_smooth_alpha  \
--layer_cascade_smooth_alpha \
--save_quant_meta \
--save_model_path outputs/tinysr_w4a8_quantized.pt
# --load_model_path outputs/tinysr_w4a8_quantized.pt  # skip calibration, load saved model
# --svdq_smooth_alpha 0.55  \
# --quant_config /data/disk2/xby/TinySR/train/layer_quant_config_example.json  \
# --calib_input_dir dataset/StableSR_testsets/DrealSRVal_crop128/test_LR \
# --calib_cache outputs/calib_cache.pt  \