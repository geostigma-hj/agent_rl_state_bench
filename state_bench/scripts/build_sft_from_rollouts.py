"""Build verl multiturn SFT data from STATE-Bench rollout trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from state_bench.domains.customer_support.tools import TOOL_SCHEMAS


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _clean_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if value is not None}


def _assistant_messages(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content") or ""
    tool_responses: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls") or []
    clean_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        result = call.get("result") or {}
        if isinstance(result, dict) and result.get("error"):
            continue
        name = call.get("name")
        arguments = _clean_arguments(call.get("arguments") or {})
        if not name:
            continue
        clean_calls.append({"name": name, "arguments": arguments})
        tool_responses.append({"role": "tool", "content": _json_text(call.get("result", {}))})

    if not clean_calls:
        return [{"role": "assistant", "content": content}]

    # STATE-Bench trajectories store a whole agent turn as one record: the model
    # may have made tool calls internally and then replied to the user. For SFT
    # with chat-template tool calling, preserve causal order explicitly.
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": "", "tool_calls": clean_calls}]
    messages.extend(tool_responses)
    if content.strip():
        messages.append({"role": "assistant", "content": content})
    return messages


def convert_conversation(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in conversation:
        role = str(message.get("role", "")).lower()
        if role == "assistant":
            messages.extend(_assistant_messages(message))
        elif role in {"system", "user"}:
            messages.append({"role": role, "content": message.get("content") or ""})
    while messages and messages[-1]["role"] == "user" and messages[-1]["content"].strip() == "[TASK_DONE]":
        messages.pop()
    return messages


def trajectory_passes(traj: dict[str, Any], *, require_task_completion: bool) -> bool:
    if require_task_completion and traj.get("task_completion_pass") != 1:
        return False
    if traj.get("state_requirements_met") != 1:
        return False
    if traj.get("task_requirements_met") != 1:
        return False
    if traj.get("error"):
        return False
    return True


def build_rows(
    rollout_dir: Path,
    *,
    require_task_completion: bool,
    include_tools: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in sorted(rollout_dir.glob("*.json")):
        traj = json.loads(path.read_text())
        tid = traj.get("task_id") or path.stem
        keep = trajectory_passes(traj, require_task_completion=require_task_completion)
        record = {
            "task_id": tid,
            "path": str(path),
            "task_completion_pass": traj.get("task_completion_pass"),
            "state_requirements_met": traj.get("state_requirements_met"),
            "task_requirements_met": traj.get("task_requirements_met"),
            "turns": traj.get("turns"),
            "tool_calls": traj.get("tool_calls"),
        }
        if not keep:
            rejected.append(record)
            continue
        row: dict[str, Any] = {
            "task_id": tid,
            "trajectory_path": str(path),
            "messages": convert_conversation(traj.get("conversation") or []),
            "enable_thinking": False,
        }
        if include_tools:
            row["tools"] = TOOL_SCHEMAS
        rows.append(row)
    return rows, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        required=True,
        action="append",
        help="Directory containing scored trajectory JSONs. Repeat to merge multiple rollout directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for train/val parquet outputs.")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--no-tools", action="store_true", help="Do not include tool schemas column.")
    parser.add_argument(
        "--allow-partial-pass",
        action="store_true",
        help="Keep trajectories with state/task pass even if task_completion_pass is missing.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rollout_dir in args.rollout_dir:
        rollout_rows, rollout_rejected = build_rows(
            rollout_dir,
            require_task_completion=not args.allow_partial_pass,
            include_tools=not args.no_tools,
        )
        rows.extend(rollout_rows)
        rejected.extend(rollout_rejected)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_n = max(1, int(round(len(rows) * args.val_ratio))) if len(rows) > 1 else 0
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    pd.DataFrame(train_rows).to_parquet(args.output_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(args.output_dir / "val.parquet", index=False)
    manifest = {
        "rollout_dirs": [str(path) for path in args.rollout_dir],
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "accepted_count": len(rows),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "include_tools": not args.no_tools,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ["train_count", "val_count", "accepted_count", "rejected_count"]}))


if __name__ == "__main__":
    main()
