CUDA_VISIBLE_DEVICES=7 python test/benchmark.py \
  --model t \
  --pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
  --lora_dir /data/disk2/xby/TinySR/checkpoint/pyramid-stage1/checkpoint-56001 \
  --vae_path="checkpoint/vae/separable" \
  --embedding_dir dataset/default/ \
  --batch_size 64 \
  --num_iterations 100 \
  --mixed_precision fp16
  # --lora_dir="checkpoint/tinysr" \