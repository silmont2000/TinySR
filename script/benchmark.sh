CUDA_VISIBLE_DEVICES=1 python test/benchmark.py \
  --model p \
  --pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
  --lora_dir /data/disk2/xby/TinySR/checkpoint/pyramid-stage1/checkpoint-240001 \
  --vae_path="checkpoint/vae/separable" \
  --embedding_dir dataset/default/ \
  --batch_size 64 \
  --eval_size 512 \
  --num_iterations 50 \
  --mixed_precision fp16