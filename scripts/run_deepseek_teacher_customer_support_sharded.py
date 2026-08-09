#!/usr/bin/env python3
"""Launch DeepSeek teacher rollouts through the official STATE-Bench harness in shards."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def _task_ids(tasks_dir: Path) -> list[str]:
    return sorted(path.stem for path in tasks_dir.glob("*.json"))


def _parse_task_ids(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _completed_ids(run_dir: Path) -> set[str]:
    if not run_dir.exists():
        return set()
    return {path.stem for path in run_dir.glob("*.json")}


def _chunks(items: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, min(n_chunks, len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _configure_deepseek_env(config_path: Path, env: dict[str, str]) -> dict[str, str]:
    env.setdefault("DEEPSEEK_CONFIG_JSON", str(config_path))
    env.setdefault("DEEPSEEK_AGENT_MODEL", "deepseek-v4-flash")
    env.setdefault("DEEPSEEK_AGENT_THINKING", "disabled")
    env.setdefault("STATE_BENCH_EVAL_PROVIDER", "deepseek")
    env.setdefault("STATE_BENCH_EVAL_MODEL", "deepseek-v4-flash")
    env.setdefault("STATE_BENCH_EVAL_DEPLOYMENTS", env["STATE_BENCH_EVAL_MODEL"])
    env.setdefault("STATE_BENCH_SIMULATOR_TEMPERATURE", "0.2")

    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        if config.get("base_url"):
            env.setdefault("DEEPSEEK_AGENT_BASE_URL", config["base_url"])
            env.setdefault("STATE_BENCH_EVAL_BASE_URL", config["base_url"])
        if config.get("api_key"):
            env.setdefault("DEEPSEEK_AGENT_API_KEY", config["api_key"])
            env.setdefault("STATE_BENCH_EVAL_API_KEY", config["api_key"])

    missing = [
        name
        for name in (
            "DEEPSEEK_AGENT_BASE_URL",
            "DEEPSEEK_AGENT_API_KEY",
            "STATE_BENCH_EVAL_BASE_URL",
            "STATE_BENCH_EVAL_API_KEY",
        )
        if not env.get(name)
    ]
    if missing:
        raise SystemExit(f"Missing required setting(s): {', '.join(missing)}")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard synthetic customer_support teacher rollouts.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tasks-dir", default=Path("synthetic_data/customer_support/tasks"), type=Path)
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs to run.")
    parser.add_argument("--tasks-file", default=None, type=Path, help="Text file with one task ID per line.")
    parser.add_argument("--run-idx", default=1, type=int)
    parser.add_argument("--shards", default=12, type=int)
    parser.add_argument("--workers-per-shard", default=8, type=int)
    parser.add_argument("--retry-attempts", default=3, type=int)
    parser.add_argument("--agent-model-name", default="deepseek-v4-flash")
    parser.add_argument("--deepseek-config-json", default=Path("/home/hj/shixi/deepseek.json"), type=Path)
    parser.add_argument("--poll-seconds", default=30, type=int)
    args = parser.parse_args()
    child_env = _configure_deepseek_env(args.deepseek_config_json, os.environ.copy())

    if args.tasks and args.tasks_file:
        raise SystemExit("--tasks and --tasks-file are mutually exclusive")
    if args.tasks:
        all_ids = _parse_task_ids(args.tasks)
    elif args.tasks_file:
        all_ids = [
            line.strip()
            for line in args.tasks_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        all_ids = _task_ids(args.tasks_dir)
    if not all_ids:
        raise SystemExit(f"No task JSON files found under {args.tasks_dir}")
    known_ids = set(_task_ids(args.tasks_dir))
    missing_task_files = [task_id for task_id in all_ids if task_id not in known_ids]
    if missing_task_files:
        preview = ", ".join(missing_task_files[:20])
        raise SystemExit(f"Task file(s) not found under {args.tasks_dir}: {preview}")

    run_dir = args.output_dir / f"run{args.run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    completed_before = _completed_ids(run_dir)
    missing = [task_id for task_id in all_ids if task_id not in completed_before]
    print(
        f"Teacher rollout shard launcher: {len(completed_before)}/{len(all_ids)} already complete; "
        f"{len(missing)} missing.",
        flush=True,
    )
    if not missing:
        return 0

    processes: list[tuple[int, subprocess.Popen[bytes], Path]] = []
    for shard_idx, shard_tasks in enumerate(_chunks(missing, args.shards), start=1):
        log_path = logs_dir / f"run{args.run_idx}_shard{shard_idx:02d}.log"
        cmd = [
            sys.executable,
            "-m",
            "state_bench.scripts.run_batch",
            "--domain",
            "customer_support",
            "--tasks-dir",
            str(args.tasks_dir),
            "--split",
            "all",
            "--tasks",
            ",".join(shard_tasks),
            "--agent-class",
            "DeepSeekToolCallingAgent",
            "--agent-client-class",
            "DeepSeekCompletionClient",
            "--agent-model-name",
            args.agent_model_name,
            "--num-runs",
            "1",
            "--num-runs-idx-start",
            str(args.run_idx),
            "--num-workers",
            str(args.workers_per_shard),
            "--output-dir",
            str(args.output_dir),
            "--no-score",
            "--retry-attempts",
            str(args.retry_attempts),
        ]
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=child_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.close()
        processes.append((shard_idx, process, log_path))
        print(f"Started shard {shard_idx:02d}: {len(shard_tasks)} tasks, pid={process.pid}, log={log_path}", flush=True)

    pids_path = logs_dir / f"run{args.run_idx}_shard_pids.txt"
    pids_path.write_text("\n".join(f"{idx}\t{proc.pid}\t{log}" for idx, proc, log in processes) + "\n")

    last_count = -1
    while True:
        complete = len(_completed_ids(run_dir))
        if complete != last_count:
            print(f"Progress: {complete}/{len(all_ids)} trajectories written", flush=True)
            last_count = complete

        alive = [(idx, proc, log) for idx, proc, log in processes if proc.poll() is None]
        if not alive:
            break
        time.sleep(max(5, args.poll_seconds))

    failed = [(idx, proc.returncode, log) for idx, proc, log in processes if proc.returncode != 0]
    complete = len(_completed_ids(run_dir))
    print(f"Final progress: {complete}/{len(all_ids)} trajectories written", flush=True)
    if failed:
        for idx, returncode, log in failed:
            print(f"Shard {idx:02d} failed with exit code {returncode}; see {log}", flush=True)
        return 1
    if complete < len(all_ids):
        print(f"Shard processes finished but {len(all_ids) - complete} tasks are still missing.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
