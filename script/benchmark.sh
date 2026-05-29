CUDA_VISIBLE_DEVICES=2 python test/benchmark.py \
  --model p \
  --pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
  --lora_dir /data/disk2/xby/TinySR/checkpoint/checkpoint-126001 \
  --vae_path="checkpoint/vae/separable" \
  --embedding_dir dataset/default/ \
  --batch_size 16 \
  --num_iterations 100 \
  --mixed_precision fp16
  # --lora_dir="checkpoint/tinysr" \