"""Build a verl RL prompt dataset from STATE-Bench task JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from state_bench.domain import get_domain_config
from state_bench.schemas import TaskDefinition


def _parse_task_ids(raw_value: str | None, tasks_dir: Path) -> list[str]:
    if raw_value:
        return [part.strip() for part in raw_value.split(",") if part.strip()]
    return sorted(path.stem for path in tasks_dir.glob("*.json"))


def _normalize_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    if "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _row_for_task(domain_name: str, tasks_dir: Path, task_id: str, split: str, index: int) -> dict[str, Any]:
    domain = get_domain_config(domain_name)
    task_path = tasks_dir / f"{task_id}.json"
    task = TaskDefinition.load(task_path)
    system_prompt = domain.agent_system_prompt.format(now=task.now, user_id=task.user_id)
    return {
        "data_source": f"state_bench/{domain_name}",
        "agent_name": "state_bench_customer_support",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.opening_message},
        ],
        "ability": "agent_rl",
        "reward_model": {"style": "state_bench_reward_v1", "ground_truth": task.task_id},
        "extra_info": {
            "split": split,
            "index": index,
            "domain": domain_name,
            "task_id": task.task_id,
            "task_path": str(task_path),
            "tasks_dir": str(tasks_dir),
            "task_env_path": task.task_env_path,
            "user_id": task.user_id,
            "task_type": task.task_type,
            "tool_selection": [tool["function"]["name"] for tool in map(_normalize_tool_schema, domain.tool_schemas)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verl RL parquet data from STATE-Bench tasks.")
    parser.add_argument("--domain", default="customer_support")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs. Defaults to all tasks.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    task_ids = _parse_task_ids(args.tasks, args.tasks_dir)
    if not task_ids:
        raise ValueError(f"No task JSON files found in {args.tasks_dir}")
    rows = [_row_for_task(args.domain, args.tasks_dir, task_id, args.split, idx) for idx, task_id in enumerate(task_ids)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))

    manifest_path = args.manifest or (args.output.parent / "manifest.json")
    manifest = {
        "domain": args.domain,
        "tasks_dir": str(args.tasks_dir),
        "split": args.split,
        "num_tasks": len(task_ids),
        "task_ids": task_ids,
        "output": str(args.output),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("domain", "split", "num_tasks", "output")}, ensure_ascii=False))
    print(f"Saved: {args.output}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
