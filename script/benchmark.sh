CUDA_VISIBLE_DEVICES=1 python test/benchmark.py \
  --model t \
  --pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
  --lora_dir="/data/disk2/xby/TinySR/checkpoint/tinysr" \
  --vae_path="checkpoint/vae/separable" \
  --embedding_dir dataset/default/ \
  --batch_size 30 \
  --eval_size 512 \
  --num_iterations 50 \
  --mixed_precision fp16
  # --lora_dir /data/disk2/xby/TinySR/checkpoint/pyramid-stage1/checkpoint-501 \