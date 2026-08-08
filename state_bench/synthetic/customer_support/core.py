"""Shared helpers for customer_support synthetic generators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.paths import PACKAGE_ROOT
from state_bench.schemas import TaskDefinition
from state_bench.synthetic.customer_support.checker import replay_gold_tool_plan, requirements_from_state_diff
from state_bench.synthetic.customer_support.llm import OpenAICompatibleJsonClient


def repo_root() -> Path:
    return PACKAGE_ROOT.parent


def load_seed(seed_task_id: str) -> tuple[TaskDefinition, CSEnvironmentData]:
    root = repo_root() / "state_bench" / "domains" / "customer_support"
    return (
        TaskDefinition.load(root / "tasks" / f"{seed_task_id}.json"),
        CSEnvironmentData.load(root / "task_envs" / f"{seed_task_id}.json"),
    )


def output_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    tasks_dir = output_root / "tasks"
    envs_dir = output_root / "task_envs"
    metadata_dir = output_root / "metadata"
    for directory in (tasks_dir, envs_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return tasks_dir, envs_dir, metadata_dir


def finalize_task(
    *,
    task: TaskDefinition,
    env_data: CSEnvironmentData,
    metadata: dict[str, Any],
    output_root: Path,
    rewrite: bool,
) -> dict[str, Path]:
    tasks_dir, envs_dir, metadata_dir = output_dirs(output_root)
    if rewrite:
        rewrite_language(task, metadata, OpenAICompatibleJsonClient.deepseek_v4_flash())

    env_path = envs_dir / f"{task.task_id}.json"
    task_path = tasks_dir / f"{task.task_id}.json"
    metadata_path = metadata_dir / f"{task.task_id}.json"
    task.task_env_path = str(Path("synthetic_data") / "customer_support" / "task_envs" / env_path.name)
    env_data.save(env_path)
    write_json(task_path, task.to_dict())
    write_json(metadata_path, metadata)
    return {"task": task_path, "env": env_path, "metadata": metadata_path}


def build_state_requirements(env_data: CSEnvironmentData, now: str, gold_tool_plan: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_diff = replay_gold_tool_plan(env_data, now, gold_tool_plan)
    return requirements_from_state_diff(state_diff), state_diff.to_dict()


def rewrite_language(task: TaskDefinition, metadata: dict[str, Any], client: OpenAICompatibleJsonClient) -> None:
    from state_bench.schemas import UserSimulatorConfig

    system = (
        "You rewrite synthetic customer-support benchmark task language. Return JSON only. "
        "Do not change any IDs, dates, prices, policy-relevant facts, or expected outcome."
    )
    user = json.dumps(
        {
            "opening_message": task.opening_message,
            "task_summary": task.task_summary,
            "user_simulator": task.user_simulator.to_dict(),
            "constraints": [
                "Keep all IDs exactly unchanged.",
                "Keep the same user goal and final outcome.",
                "Make the language natural but unambiguous.",
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
    rewritten = client.complete_json(system=system, user=user)
    task.opening_message = rewritten.get("opening_message", task.opening_message)
    task.task_summary = rewritten.get("task_summary", task.task_summary)
    sim = rewritten.get("user_simulator")
    if isinstance(sim, dict):
        task.user_simulator = UserSimulatorConfig.from_dict({**task.user_simulator.to_dict(), **sim})
    metadata["language_rewritten"] = True
    metadata["language_rewriter_model"] = client.model


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
