#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
VERL_ROOT="${VERL_ROOT:-/home/hj/shixi/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/hj/shixi/.conda/envs/verl-rl-base2/bin/python}"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export STATE_BENCH_EVAL_PROVIDER="${STATE_BENCH_EVAL_PROVIDER:-deepseek}"
export STATE_BENCH_EVAL_MODEL="${STATE_BENCH_EVAL_MODEL:-deepseek-v4-flash}"
export STATE_BENCH_EVAL_DEPLOYMENTS="${STATE_BENCH_EVAL_DEPLOYMENTS:-$STATE_BENCH_EVAL_MODEL}"
export STATE_BENCH_SIMULATOR_TEMPERATURE="${STATE_BENCH_SIMULATOR_TEMPERATURE:-0.2}"
export STATE_BENCH_EVAL_MAX_TOKENS="${STATE_BENCH_EVAL_MAX_TOKENS:-1024}"
export STATE_BENCH_DEEPSEEK_THINKING="${STATE_BENCH_DEEPSEEK_THINKING:-disabled}"

eval_config_json="${STATE_BENCH_EVAL_CONFIG_JSON:-/home/hj/shixi/deepseek.json}"
if [[ -f "$eval_config_json" ]]; then
  eval_exports="$(
    EVAL_CONFIG_JSON="$eval_config_json" "$PYTHON_BIN" - <<'PY'
import json
import os
import shlex

with open(os.environ["EVAL_CONFIG_JSON"], encoding="utf-8") as f:
    data = json.load(f)
if not os.environ.get("STATE_BENCH_EVAL_BASE_URL"):
    print("export STATE_BENCH_EVAL_BASE_URL=" + shlex.quote(data["base_url"]))
if not os.environ.get("STATE_BENCH_EVAL_API_KEY"):
    print("export STATE_BENCH_EVAL_API_KEY=" + shlex.quote(data["api_key"]))
PY
  )"
  if [[ -n "$eval_exports" ]]; then
    eval "$eval_exports"
  fi
fi

missing=()
[[ -n "${STATE_BENCH_EVAL_BASE_URL:-}" ]] || missing+=("STATE_BENCH_EVAL_BASE_URL")
[[ -n "${STATE_BENCH_EVAL_API_KEY:-}" ]] || missing+=("STATE_BENCH_EVAL_API_KEY")
if (( ${#missing[@]} )); then
  printf 'Missing required setting(s): %s\n' "${missing[*]}" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/customer_support/qwen3_4b_sft_lora_r16_a32_round123_toolsplit_clean_main8192_lr1e4_b20/global_step_100_hf_lora_merged}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/rl/customer_support_stage1_synthetic32/train.parquet}"
VAL_FILE="${VAL_FILE:-$TRAIN_FILE}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}"

ACTOR_LR="${ACTOR_LR:-3e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.65}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS:-8}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.8}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-${MAX_TOOL_ROUNDS:-10}}"
AGENT_TURN_MAX_TOKENS="${AGENT_TURN_MAX_TOKENS:-1024}"
export STATE_BENCH_RL_AGENT_TURN_MAX_TOKENS="$AGENT_TURN_MAX_TOKENS"

PROJECT_NAME="${PROJECT_NAME:-state_bench_customer_support_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_sft_step100_lora_grpo_stage1_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
LOGGER="${LOGGER:-[\"console\"]}"

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=True
  data.truncation=error
)

MODEL=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank="$LORA_RANK"
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF"
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
  actor_rollout_ref.actor.entropy_from_logits_chunk_size=256
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=256
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL"
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER"
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE"
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P"
  actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K"
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=True
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
  actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_AGENT_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=4
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024
  actor_rollout_ref.rollout.multi_turn.format=hermes
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable
  actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_NUM_WORKERS"
  actor_rollout_ref.rollout.agent.default_agent_loop=state_bench_customer_support
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/configs/verl_state_bench_agent_loop.yaml"
)

REF=(
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
  actor_rollout_ref.ref.entropy_from_logits_chunk_size=256
  actor_rollout_ref.ref.use_torch_compile=False
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
  actor_rollout_ref.ref.fsdp_config.entropy_from_logits_with_chunking=True
  actor_rollout_ref.ref.fsdp_config.entropy_from_logits_chunk_size=256
  actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
  trainer.balance_batch=True
  trainer.logger="$LOGGER"
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.n_gpus_per_node="$NGPUS_PER_NODE"
  trainer.nnodes=1
  trainer.val_before_train=False
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
  trainer.default_local_dir="$REPO_ROOT/checkpoints/customer_support/rl/$EXPERIMENT_NAME"
  trainer.rollout_data_dir="$REPO_ROOT/outputs/rl_train_rollouts/$EXPERIMENT_NAME"
)

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${REF[@]}" \
  "${TRAINER[@]}" \
  "$@"
