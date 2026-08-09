#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export OPENAI_COMPAT_AGENT_BASE_URLS="${OPENAI_COMPAT_AGENT_BASE_URLS:-http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1,http://127.0.0.1:8006/v1}"
export OPENAI_COMPAT_AGENT_MODEL="${OPENAI_COMPAT_AGENT_MODEL:-Qwen3-4B-Instruct-2507}"
export OPENAI_COMPAT_AGENT_API_KEY="${OPENAI_COMPAT_AGENT_API_KEY:-EMPTY}"

export STATE_BENCH_EVAL_PROVIDER="${STATE_BENCH_EVAL_PROVIDER:-deepseek}"
export STATE_BENCH_EVAL_MODEL="${STATE_BENCH_EVAL_MODEL:-deepseek-v4-flash}"
export STATE_BENCH_EVAL_DEPLOYMENTS="${STATE_BENCH_EVAL_DEPLOYMENTS:-$STATE_BENCH_EVAL_MODEL}"
export STATE_BENCH_SIMULATOR_TEMPERATURE="${STATE_BENCH_SIMULATOR_TEMPERATURE:-0.2}"

eval_config_json="${STATE_BENCH_EVAL_CONFIG_JSON:-/home/hj/shixi/deepseek.json}"
if [[ -f "$eval_config_json" ]]; then
  eval_exports="$(
    EVAL_CONFIG_JSON="$eval_config_json" .venv/bin/python - <<'PY'
import json
import os
import shlex

data = json.load(open(os.environ["EVAL_CONFIG_JSON"]))
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

split="${SPLIT:-test}"
tasks_dir="${TASKS_DIR:-}"
score_split="${SCORE_SPLIT:-$([[ -n "${TASKS:-}" ]] && printf all || printf %s "$split")}"
task_limit="${TASK_LIMIT:-}"
num_runs="${NUM_RUNS:-1}"
num_runs_idx_start="${NUM_RUNS_IDX_START:-1}"
num_workers="${NUM_WORKERS:-7}"
model_label="${MODEL_LABEL:-${OPENAI_COMPAT_AGENT_MODEL//\//_}}"
output_dir="${OUTPUT_DIR:-outputs/customer_support_${model_label}_native_${split}}"

if [[ -z "${TASKS:-}" && -n "$task_limit" ]]; then
  TASKS="$(
    SPLIT="$split" TASK_LIMIT="$task_limit" .venv/bin/python - <<'PY'
import os
from state_bench.protocol import load_default_protocol, load_split_task_ids

protocol = load_default_protocol()
task_ids = load_split_task_ids("customer_support", os.environ["SPLIT"], protocol.split_version)
print(",".join(task_ids[: int(os.environ["TASK_LIMIT"])]))
PY
  )"
fi

args=(
  -m state_bench.scripts.run_batch
  --domain customer_support
  --split "$([[ -n "${TASKS:-}" ]] && printf all || printf %s "$split")"
  --agent-class OpenAICompatibleToolCallingAgent
  --agent-client-class OpenAICompatibleToolClient
  --agent-model-name "$OPENAI_COMPAT_AGENT_MODEL"
  --num-runs "$num_runs"
  --num-runs-idx-start "$num_runs_idx_start"
  --num-workers "$num_workers"
  --output-dir "$output_dir"
)

if [[ -n "$tasks_dir" ]]; then
  args+=(--tasks-dir "$tasks_dir")
fi

if [[ -n "${TASKS:-}" ]]; then
  args+=(--tasks "$TASKS")
fi

if [[ "${NO_UX_SCORE:-0}" == "1" ]]; then
  args+=(--no-ux-score)
fi

.venv/bin/python "${args[@]}"

score_args=(
  -m state_bench.scripts.compute_metrics
  --domain customer_support
  --split "$score_split"
  --results-dir "$output_dir"
  --num-runs "$num_runs"
  --num-runs-idx-start "$num_runs_idx_start"
  --save-filepath "$output_dir/metrics.json"
)

if [[ -n "${TASKS:-}" ]]; then
  score_args+=(--ignore-missing-runs)
fi

.venv/bin/python "${score_args[@]}"
