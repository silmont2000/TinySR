export MODEL_NAME="your model";
export TSDSR="your teacher";
export DINO_MODEL_PATH="dinov2 path";
export LORA_DIR="resume from stage1";
export OUTPUT_DIR="checkpoint/train-stage2/";
export OUTPUT_LOG="logs/stage2.log";
export LOG_NAME="train-stage2";
nohup accelerate launch  --config_file config/config.yaml  --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --main_process_port 52341 --mixed_precision="fp16" train/train_stage2.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --lora_dir=$LORA_DIR \
  --teacher_model_name_or_path=$TSDSR \
  --DINO_v2_pretrained_model_path=$DINO_MODEL_PATH \
  --train_batch_size=12 \
  --num_train_epochs=200 --checkpointing_steps=1000 \
  --learning_rate=1e-06 --learning_rate_discrimitor=5e-6 --validation_steps 5000  --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=3000 \
  --seed=80 \
  --output_dir=$OUTPUT_DIR \
  --max_train_steps=50000 \
  --gradient_accumulation_steps=1 \
  --resume_from_checkpoint="latest" \
  --report_to="wandb" --log_code --log_name=$LOG_NAME > $OUTPUT_LOG 2>&1 &





