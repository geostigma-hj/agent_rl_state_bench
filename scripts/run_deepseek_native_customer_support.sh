#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export DEEPSEEK_CONFIG_JSON="${DEEPSEEK_CONFIG_JSON:-/home/hj/shixi/deepseek.json}"
export DEEPSEEK_AGENT_MODEL="${DEEPSEEK_AGENT_MODEL:-deepseek-v4-flash}"
export DEEPSEEK_AGENT_THINKING="${DEEPSEEK_AGENT_THINKING:-disabled}"

export STATE_BENCH_EVAL_PROVIDER="${STATE_BENCH_EVAL_PROVIDER:-deepseek}"
export STATE_BENCH_EVAL_MODEL="${STATE_BENCH_EVAL_MODEL:-deepseek-v4-flash}"
export STATE_BENCH_EVAL_DEPLOYMENTS="${STATE_BENCH_EVAL_DEPLOYMENTS:-$STATE_BENCH_EVAL_MODEL}"
export STATE_BENCH_SIMULATOR_TEMPERATURE="${STATE_BENCH_SIMULATOR_TEMPERATURE:-0.2}"

if [[ -f "$DEEPSEEK_CONFIG_JSON" ]]; then
  eval_exports="$(
    EVAL_CONFIG_JSON="$DEEPSEEK_CONFIG_JSON" .venv/bin/python - <<'PY'
import json
import os
import shlex

data = json.load(open(os.environ["EVAL_CONFIG_JSON"]))
if not os.environ.get("DEEPSEEK_AGENT_BASE_URL"):
    print("export DEEPSEEK_AGENT_BASE_URL=" + shlex.quote(data["base_url"]))
if not os.environ.get("DEEPSEEK_AGENT_API_KEY"):
    print("export DEEPSEEK_AGENT_API_KEY=" + shlex.quote(data["api_key"]))
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
[[ -n "${DEEPSEEK_AGENT_BASE_URL:-}" ]] || missing+=("DEEPSEEK_AGENT_BASE_URL")
[[ -n "${DEEPSEEK_AGENT_API_KEY:-}" ]] || missing+=("DEEPSEEK_AGENT_API_KEY")
[[ -n "${STATE_BENCH_EVAL_BASE_URL:-}" ]] || missing+=("STATE_BENCH_EVAL_BASE_URL")
[[ -n "${STATE_BENCH_EVAL_API_KEY:-}" ]] || missing+=("STATE_BENCH_EVAL_API_KEY")
if (( ${#missing[@]} )); then
  printf 'Missing required setting(s): %s\n' "${missing[*]}" >&2
  exit 2
fi

split="${SPLIT:-test}"
task_limit="${TASK_LIMIT:-}"
num_runs="${NUM_RUNS:-1}"
num_workers="${NUM_WORKERS:-1}"
model_label="${MODEL_LABEL:-${DEEPSEEK_AGENT_MODEL//\//_}}"
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
  --agent-class DeepSeekToolCallingAgent
  --agent-client-class DeepSeekCompletionClient
  --agent-model-name "$DEEPSEEK_AGENT_MODEL"
  --num-runs "$num_runs"
  --num-workers "$num_workers"
  --output-dir "$output_dir"
)

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
  --split "$split"
  --results-dir "$output_dir"
  --num-runs "$num_runs"
  --save-filepath "$output_dir/metrics.json"
)

if [[ -n "${TASKS:-}" ]]; then
  score_args+=(--ignore-missing-runs)
fi

.venv/bin/python "${score_args[@]}"
