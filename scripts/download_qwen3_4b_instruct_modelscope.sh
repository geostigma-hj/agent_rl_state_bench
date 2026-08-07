#!/usr/bin/env bash
set -euo pipefail

model_id="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
local_dir="${LOCAL_DIR:-/mnt/models/Qwen3-4B-Instruct-2507}"
conda_python="${VERL_CONDA_PYTHON:-/home/hj/shixi/.conda/envs/verl-rl-base2/bin/python}"

export MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-2}"

"$conda_python" - <<PY
from modelscope import snapshot_download

path = snapshot_download(
    "${model_id}",
    local_dir="${local_dir}",
    max_workers=8,
)
print(path)
PY
