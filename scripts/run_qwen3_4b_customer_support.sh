#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export STATE_BENCH_EVAL_PROVIDER="${STATE_BENCH_EVAL_PROVIDER:-stepfun}"
export STATE_BENCH_EVAL_MODEL="${STATE_BENCH_EVAL_MODEL:-step-3.7-flash}"
export STATE_BENCH_EVAL_DEPLOYMENTS="${STATE_BENCH_EVAL_DEPLOYMENTS:-$STATE_BENCH_EVAL_MODEL}"
export STATE_BENCH_SIMULATOR_TEMPERATURE="${STATE_BENCH_SIMULATOR_TEMPERATURE:-0.2}"

if [[ "$STATE_BENCH_EVAL_PROVIDER" == "stepfun" && -f /home/hj/shixi/step.json ]]; then
  step_exports="$(
    .venv/bin/python - <<'PY'
import json
import os
import shlex

data = json.load(open("/home/hj/shixi/step.json"))
if not os.environ.get("STATE_BENCH_EVAL_BASE_URL"):
    print("export STATE_BENCH_EVAL_BASE_URL=" + shlex.quote(data["base_url"]))
if not os.environ.get("STATE_BENCH_EVAL_API_KEY"):
    print("export STATE_BENCH_EVAL_API_KEY=" + shlex.quote(data["api_key"]))
PY
  )"
  if [[ -n "$step_exports" ]]; then
    eval "$step_exports"
  fi
fi

missing=()
if [[ "$STATE_BENCH_EVAL_PROVIDER" == "stepfun" || "$STATE_BENCH_EVAL_PROVIDER" == "openai_compatible" ]]; then
  [[ -n "${STATE_BENCH_EVAL_BASE_URL:-}" ]] || missing+=("STATE_BENCH_EVAL_BASE_URL")
  [[ -n "${STATE_BENCH_EVAL_API_KEY:-}" ]] || missing+=("STATE_BENCH_EVAL_API_KEY")
  [[ -n "${STATE_BENCH_EVAL_MODEL:-}" ]] || missing+=("STATE_BENCH_EVAL_MODEL")
else
  [[ -n "${STATE_BENCH_EVAL_ENDPOINT:-}" ]] || missing+=("STATE_BENCH_EVAL_ENDPOINT")
  [[ -n "${STATE_BENCH_EVAL_DEPLOYMENTS:-}" ]] || missing+=("STATE_BENCH_EVAL_DEPLOYMENTS")
fi
if (( ${#missing[@]} )); then
  printf 'Missing required STATE-Bench locked eval setting(s): %s\n' "${missing[*]}" >&2
  printf 'Set them in state-bench/.env or /home/hj/shixi/step.json before running.\n' >&2
  exit 2
fi

export STATE_BENCH_VLLM_BASE_URL="${STATE_BENCH_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export STATE_BENCH_VLLM_MODEL="${STATE_BENCH_VLLM_MODEL:-Qwen3-4B}"
export STATE_BENCH_VLLM_MAX_TOKENS="${STATE_BENCH_VLLM_MAX_TOKENS:-2048}"
export STATE_BENCH_VLLM_TEMPERATURE="${STATE_BENCH_VLLM_TEMPERATURE:-0.0}"
export STATE_BENCH_VLLM_TOP_P="${STATE_BENCH_VLLM_TOP_P:-1.0}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

split="${SPLIT:-test}"
task_limit="${TASK_LIMIT:-}"
num_runs="${NUM_RUNS:-1}"
num_workers="${NUM_WORKERS:-1}"
model_label="${MODEL_LABEL:-${STATE_BENCH_VLLM_MODEL//\//_}}"
output_dir="${OUTPUT_DIR:-outputs/customer_support_${model_label}_${split}}"
agent_client_class="${AGENT_CLIENT_CLASS:-VLLMCompletionClient}"

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
  --agent-class JsonCompletionAgent
  --agent-client-class "$agent_client_class"
  --agent-model-name "$STATE_BENCH_VLLM_MODEL"
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

if [[ -z "${TASKS:-}" ]]; then
  .venv/bin/python -m state_bench.scripts.compute_metrics \
    --domain customer_support \
    --split "$split" \
    --results-dir "$output_dir" \
    --num-runs "$num_runs" \
    --save-filepath "$output_dir/metrics.json"
else
  .venv/bin/python -m state_bench.scripts.compute_metrics \
    --domain customer_support \
    --split "$split" \
    --ignore-missing-runs \
    --results-dir "$output_dir" \
    --num-runs "$num_runs" \
    --save-filepath "$output_dir/metrics.json"
fi
