source /opt/miniconda3/bin/activate tinysr
export MODEL_NAME="checkpoint/tinybackbone/prune-12-merge-tinysr";
export OUTPUT_DIR="checkpoint/pyramid-stage1/";
export OUTPUT_LOG="logs/pyramid-stage1.log";
export LOG_NAME="pyramid-stage1";

nohup accelerate launch --config_file config/config.yaml \
  --num_processes 5 \
  --main_process_port 52150 \
  train/train_stage1.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --teacher_model_name_or_path="dummy" \
  --train_batch_size=4 \
  --num_train_epochs=200 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=3000 \
  --seed=80 \
  --output_dir=$OUTPUT_DIR \
  --max_train_steps=150000 \
  --gradient_accumulation_steps=1 \
  --report_to="wandb" \
  --resume_from_checkpoint="latest" \
  --log_name=$LOG_NAME \
  --use_pyramid \
  --pyramid_num_blocks 4 4 4 \
  --pyramid_dims 768 1152 1536 \
  --pyramid_grid_hw 8 16 32 \
  > $OUTPUT_LOG 2>&1 &
