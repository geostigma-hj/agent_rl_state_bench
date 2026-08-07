#!/usr/bin/env bash
set -euo pipefail

model_path="${MODEL_PATH:-/mnt/models/Qwen3-4B}"
served_model_name="${SERVED_MODEL_NAME:-$(basename "$model_path")}"
port="${PORT:-8000}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
max_model_len="${MAX_MODEL_LEN:-32768}"
conda_python="${VERL_CONDA_PYTHON:-/home/hj/shixi/.conda/envs/verl-rl-base2/bin/python}"

export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"

exec "$conda_python" -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name "$served_model_name" \
  --host 127.0.0.1 \
  --port "$port" \
  --dtype bfloat16 \
  --max-model-len "$max_model_len"
