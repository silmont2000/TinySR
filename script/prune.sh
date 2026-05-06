export TSDSR="/path/to/your/model";
export OUTPUT_DIR="checkpoint/prune";
export OUTPUT_LOG="logs/prune.log";
export LOG_NAME="train-prune";
nohup accelerate launch  --config_file config/config.yaml  --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --main_process_port 52449 --mixed_precision="fp16" train/train_prune.py \
  --pretrained_model_name_or_path=$TSDSR  \
  --train_batch_size=1 \
  --num_train_epochs=200 --checkpointing_steps=10000 \
  --learning_rate=5e-5  --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=3000 \
  --seed=80 \
  --output_dir=$OUTPUT_DIR \
  --max_train_steps=100000 \
  --gradient_accumulation_steps=1 \
  --resume_from_checkpoint="latest" \
  --report_to="wandb" --log_code --log_name=$LOG_NAME > $OUTPUT_LOG 2>&1 &

