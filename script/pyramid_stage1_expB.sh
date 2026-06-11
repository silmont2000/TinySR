source /opt/miniconda3/bin/activate tinysr
export MODEL_NAME="checkpoint/tinybackbone/prune-12-merge-tinysr";
export OUTPUT_DIR="checkpoint/pyramid-stage1-expB/";
export OUTPUT_LOG="logs/pyramid-stage1-expB.log";
export LOG_NAME="pyramid-stage1";

nohup accelerate launch --config_file config/config_expB.yaml \
  --main_process_port 52151 \
  train/train_stage1.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_batch_size=64 \
  --num_train_epochs=20000 \
  --checkpointing_steps=5000 \
  --learning_rate=2e-04 \
  --checkpoints_total_limit=10 \
  --lr_scheduler="cosine_with_restarts" \
  --max_grad_norm=0.5 \
  --lr_warmup_steps=2000 \
  --seed=88 \
  --output_dir=$OUTPUT_DIR \
  --gradient_accumulation_steps=1 \
  --report_to="wandb" \
  --resume_from_checkpoint="latest" \
  --log_name=$LOG_NAME \
  > $OUTPUT_LOG 2>&1 &
  # --vr_loss_weight=0.1 \
  # --attn_temperature=4.0 \
  # --cos_loss_weight=1 \
  # --cos_qkv_loss_weight=1 \
  # --attn_loss_weight=1 \
  # --smoke \
  # --wandb_id 6z36l6ei \
  # 2>&1 | python -u script/ringlog.py $OUTPUT_LOG 30 &

# ============================================================
# expB 修改点（对比 expA / pyramid_stage1.sh）:
#   GPU:    1,2   (config_expB.yaml)    vs  7,0
#   PORT:   52151                        vs  52150
#   OUTPUT: checkpoint/pyramid-stage1-expB/  vs  checkpoint/pyramid-stage1/
#   LOG:    logs/pyramid-stage1-expB.log     vs  logs/pyramid-stage1.log
#   WANDB:  同名 pyramid-stage1（时间戳自动区分）
#
# 要改超参数，直接改上面 CLI 参数即可，无需改 Python 代码。
# 例如换 learning_rate:
#   --learning_rate=5e-04 \
# ============================================================

# pkill -f "train_stage1.py" 2>&1 || kill $(pgrep -f "train_stage1.py") 2>&1; echo "done"
