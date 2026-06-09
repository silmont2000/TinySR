export STAGE1_CKPT="/data/disk2/xby/TinySR/checkpoint/pyramid-stage1/checkpoint-100001"
export DINO_MODEL_PATH="/data/disk2/xby/models/dinov2_vitl14_reg4_pretrain.pth"
export OUTPUT_DIR="checkpoint/pyramid-stage2/"
export OUTPUT_LOG="logs/pyramid_stage2.log"
export LOG_NAME="pyramid-stage2"

nohup accelerate launch \
--config_file config/config_expB.yaml \
  --gpu_ids 7 \
  --num_processes 1 \
  --main_process_port 52341 \
  --mixed_precision="fp16" \
  train/train_stage2.py \
  --train_batch_size=24 \
  --num_train_epochs=200 \
  --checkpointing_steps=1000 \
  --learning_rate=5e-06 \
  --learning_rate_discrimitor=3e-4 \
  --validation_steps=5000 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=3000 \
  --seed=80 \
  --output_dir=$OUTPUT_DIR \
  --max_train_steps=50000 \
  --gradient_accumulation_steps=1 \
  --report_to="wandb" \
  --log_code \
  --log_name=$LOG_NAME \
  --lora_dir=$STAGE1_CKPT \
  --DINO_v2_pretrained_model_path=$DINO_MODEL_PATH \
> $OUTPUT_LOG 2>&1 &

  # --learning_rate_discrimitor=5e-6 \
# pkill -f "train_stage2.py" 2>&1 || kill $(pgrep -f "train_stage2.py") 2>&1; echo "done"
