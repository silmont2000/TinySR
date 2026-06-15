# source /opt/miniconda3/bin/activate tinysr

# Normal mode (original TinySD3Transformer2DModel)
CUDA_VISIBLE_DEVICES=1 python test/test_time.py \
  --pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
  --lora_dir="checkpoint/tinysr" \
  --vae_path="checkpoint/vae/separable" \
  --embedding_dir="dataset/default/" \
  --process_size=512 \
  --mixed_precision=fp16

# # echo "========================================"

# # Pyramid mode (TinyPyramidSD3Transformer2DModel with LoRA)
# CUDA_VISIBLE_DEVICES=1 python test/test_time.py \
#   --pyramid \
#   --pretrained_model_name_or_path="checkpoint/tinybackbone/prune-12-merge-tinysr" \
#   --lora_dir="/data/disk2/xby/TinySR/checkpoint/pyramid-stage1-expB/checkpoint-40001" \
#   --vae_path="checkpoint/vae/separable" \
#   --embedding_dir="dataset/default/" \
#   --process_size=512 \
#   --mixed_precision=fp16
