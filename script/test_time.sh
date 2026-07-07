CUDA_VISIBLE_DEVICES=1 python test/test_time.py \
  --pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
  --lora_dir="checkpoint/tinysr" \
  --vae_path="checkpoint/vae/separable" \
  --embedding_dir="dataset/default/" \
  --process_size=512 \
  --mixed_precision=fp16