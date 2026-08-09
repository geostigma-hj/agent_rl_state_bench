#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VERL_ENV="${VERL_ENV:-/home/hj/shixi/.conda/envs/verl-rl-base2}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/Qwen3-4B-Instruct-2507}"
DATA_DIR="${DATA_DIR:-data/sft/customer_support_v4_flash_teacher_round123_targeted}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"
LR="${LR:-1e-5}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"extra\"]}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-4}"
SP_SIZE="${SP_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
SAVE_PATH="${SAVE_PATH:-checkpoints/customer_support/qwen3_4b_sft_cold_start}"
PROJECT_NAME="${PROJECT_NAME:-state-bench-customer-support-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_instruct_2507_v4_flash_teacher}"
LOG_DIR="${LOG_DIR:-logs/sft}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing TRAIN_FILE: $TRAIN_FILE" >&2
  exit 2
fi
if [[ ! -f "$VAL_FILE" ]]; then
  echo "Missing VAL_FILE: $VAL_FILE" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging SFT run to: $LOG_FILE"
echo "DATA_DIR=$DATA_DIR"
echo "SAVE_PATH=$SAVE_PATH"
echo "NPROC_PER_NODE=$NPROC_PER_NODE TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU=$MICRO_BATCH_SIZE_PER_GPU"
echo "MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU MAX_LENGTH=$MAX_LENGTH ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA"
echo "TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ LR=$LR"
echo "CHECKPOINT_SAVE_CONTENTS=$CHECKPOINT_SAVE_CONTENTS MAX_CKPT_TO_KEEP=$MAX_CKPT_TO_KEEP"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$VERL_ENV/bin/torchrun" --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.use_dynamic_bsz=true \
  data.truncation=error \
  data.ignore_input_ids_mismatch=true \
  model.path="$MODEL_PATH" \
  model.use_remove_padding=true \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  engine=fsdp \
  optim.lr="$LR" \
  +model.override_config.attn_implementation="$ATTN_IMPLEMENTATION" \
  engine.ulysses_sequence_parallel_size="$SP_SIZE" \
  trainer.default_local_dir="$SAVE_PATH" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='["console"]' \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
  checkpoint.load_contents="$CHECKPOINT_SAVE_CONTENTS" \
  trainer.max_ckpt_to_keep="$MAX_CKPT_TO_KEEP" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.resume_mode=auto
