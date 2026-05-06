export MODEL_NAME="your model";
export TSDSR="your teacher";
export OUTPUT_DIR="checkpoint/train-stage1/";
export OUTPUT_LOG="logs/stage1.log";
export LOG_NAME="train-stage1";
nohup accelerate launch  --config_file config/config.yaml  --gpu_ids 0,1,2,3,4,5,6,7, --num_processes 8 --main_process_port 52149 --mixed_precision="fp16" train/train_stage1.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --teacher_model_name_or_path=$TSDSR \
  --train_batch_size=16  \
  --num_train_epochs=200 --checkpointing_steps=10000 \
  --learning_rate=1e-04  --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=3000 \
  --seed=80 \
  --output_dir=$OUTPUT_DIR \
  --max_train_steps=150000 \
  --gradient_accumulation_steps=1 \
  --report_to="wandb" --log_code --log_name=$LOG_NAME \
  --resume_from_checkpoint="latest" \ > $OUTPUT_LOG 2>&1 &





