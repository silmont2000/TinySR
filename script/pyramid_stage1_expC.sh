source /opt/miniconda3/bin/activate tinysr
export MODEL_NAME="checkpoint/tinybackbone/prune-12-merge-tinysr";
export OUTPUT_DIR="checkpoint/pyramid-stage1-expC/";
export OUTPUT_LOG="logs/pyramid-stage1-expC.log";
export LOG_NAME="pyramid-stage1";

nohup accelerate launch --config_file config/config_expC.yaml \
  --main_process_port 52152 \
  train/train_stage1.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_batch_size=64 \
  --num_train_epochs=20000 \
  --checkpointing_steps=1000 \
  --learning_rate=2e-04 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="cosine_with_restarts" \
  --max_grad_norm=1 \
  --lr_warmup_steps=2000 \
  --seed=88 \
  --output_dir=$OUTPUT_DIR \
  --gradient_accumulation_steps=1 \
  --report_to="wandb" \
  --resume_from_checkpoint="latest" \
  --log_name=$LOG_NAME \
  > $OUTPUT_LOG 2>&1 &
  # --vr_loss_weight=0.4 \
  # --attn_temperature=4.0 \
  # --cos_loss_weight=1 \
  # --cos_qkv_loss_weight=1 \
  # --attn_loss_weight=1 \
  # --smoke \
  # --wandb_id 6z36l6ei \
  # 2>&1 | python -u script/ringlog.py $OUTPUT_LOG 30 &


# pkill -f "train_stage1.py" 2>&1 || kill $(pgrep -f "train_stage1.py") 2>&1; echo "done"
