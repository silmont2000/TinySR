python test/test_tinysr.py  \
--pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
--vae_path="checkpoint/vae/separable" \
--lora_dir="/data/disk2/xby/TinySR/checkpoint/tinysr" \
--embedding_dir="dataset/default/" \
--output_dir="outputs/tinysr_test/" \
--input_dir="/data/disk2/xby/TinySR/dataset/StableSR_testsets/DrealSRVal_crop128/test_LR" \
--rank=64 \
--rank_vae=64 \
--is_use_tile=False \
--vae_decoder_tiled_size=224 \
--vae_encoder_tiled_size=1024 \
--latent_tiled_size=64 \
--latent_tiled_overlap=8 \
--device=cuda \
--seed=42 \
--upscale=4 \
--process_size=512 \
--mixed_precision=fp16 \
--align_method=adain


# python test/test_tsdsr.py \
# --pretrained_model_name_or_path="/data/disk2/xby/sd3-medium" \
# -i /data/disk2/xby/TinySR/dataset/smoke/sr_bicubic_128 \
# -o outputs/mytest \
# --lora_dir checkpoint/tsdsr \
# --embedding_dir dataset/default/ 