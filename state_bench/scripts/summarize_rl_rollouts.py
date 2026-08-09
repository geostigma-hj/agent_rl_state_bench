"""Summarize STATE-Bench rollouts into reward records for RL sanity runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from state_bench.rl.reward import compute_reward_v1_from_trajectory


def _iter_run_files(results_dir: Path, run_start: int, num_runs: int) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for run_idx in range(run_start, run_start + num_runs):
        run_dir = results_dir / f"run{run_idx}"
        if not run_dir.exists():
            continue
        files.extend((run_idx, path) for path in sorted(run_dir.glob("*.json")))
    return files


def _conversation_tool_calls(traj: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in traj.get("conversation", []) or []:
        msg_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if isinstance(msg_calls, list):
            calls.extend(call for call in msg_calls if isinstance(call, dict))
    return calls


def _tool_error_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for call in tool_calls:
        result = call.get("result")
        if isinstance(result, dict) and "error" in result:
            names.append(str(call.get("name", "<unknown>")))
    return names


def _reward_record(run_idx: int, path: Path) -> dict[str, Any]:
    traj = json.loads(path.read_text(encoding="utf-8"))
    tool_errors = int(traj.get("tool_errors") or 0)
    redundant_calls = int(traj.get("redundant_calls") or 0)
    reward_parts = compute_reward_v1_from_trajectory(traj)
    tool_calls = _conversation_tool_calls(traj)
    return {
        "run_idx": run_idx,
        "task_id": traj.get("task_id") or path.stem,
        "state_reward": reward_parts["state_reward"],
        "reward_v1": reward_parts["reward_v1"],
        "tool_error_penalty": reward_parts["tool_error_penalty"],
        "redundant_penalty": reward_parts["redundant_penalty"],
        "state_requirements_met": traj.get("state_requirements_met"),
        "task_requirements_met": traj.get("task_requirements_met"),
        "task_completion_pass": traj.get("task_completion_pass"),
        "turns": int(traj.get("turns") or 0),
        "tool_calls": int(traj.get("tool_calls") or 0),
        "tool_errors": tool_errors,
        "redundant_calls": redundant_calls,
        "tool_error_names": ",".join(_tool_error_names(tool_calls)),
        "trajectory_path": str(path),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_rollouts": 0}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["task_id"]), []).append(record)
    task_reward_stds = [
        pstdev(float(item["reward_v1"]) for item in items) if len(items) > 1 else 0.0 for items in grouped.values()
    ]
    return {
        "num_rollouts": len(records),
        "num_tasks": len({r["task_id"] for r in records}),
        "mean_reward_v1": round(mean(float(r["reward_v1"]) for r in records), 4),
        "mean_state_reward": round(mean(float(r["state_reward"]) for r in records), 4),
        "task_completion_pass_rate": round(
            mean(1.0 if r["task_completion_pass"] == 1 else 0.0 for r in records), 4
        ),
        "state_requirements_pass_rate": round(
            mean(1.0 if r["state_requirements_met"] == 1 else 0.0 for r in records), 4
        ),
        "task_requirements_pass_rate": round(
            mean(1.0 if r["task_requirements_met"] == 1 else 0.0 for r in records), 4
        ),
        "mean_turns": round(mean(int(r["turns"]) for r in records), 2),
        "mean_tool_calls": round(mean(int(r["tool_calls"]) for r in records), 2),
        "mean_tool_errors": round(mean(int(r["tool_errors"]) for r in records), 2),
        "mean_redundant_calls": round(mean(int(r["redundant_calls"]) for r in records), 2),
        "mean_task_reward_std": round(mean(task_reward_stds), 4),
        "variable_reward_task_count": sum(1 for value in task_reward_stds if value > 0),
    }


def _task_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["task_id"]), []).append(record)
    summaries: list[dict[str, Any]] = []
    for task_id, items in sorted(grouped.items()):
        rewards = [float(item["reward_v1"]) for item in items]
        completions = [1.0 if item["task_completion_pass"] == 1 else 0.0 for item in items]
        tool_errors = [int(item["tool_errors"]) for item in items]
        summaries.append(
            {
                "task_id": task_id,
                "num_rollouts": len(items),
                "mean_reward_v1": round(mean(rewards), 4),
                "std_reward_v1": round(pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
                "task_completion_rate": round(mean(completions), 4),
                "mean_tool_errors": round(mean(tool_errors), 2),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize STATE-Bench rollout rewards for RL sanity checks.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--num-runs-idx-start", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    records = [_reward_record(run_idx, path) for run_idx, path in _iter_run_files(args.results_dir, args.num_runs_idx_start, args.num_runs)]
    summary = _summary(records)
    payload = {"summary": summary, "task_summaries": _task_summaries(records), "records": records}

    output_json = args.output_json or (args.results_dir / "rl_reward_summary.json")
    output_csv = args.output_csv or (args.results_dir / "rl_reward_records.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if records:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_json}")
    if records:
        print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
