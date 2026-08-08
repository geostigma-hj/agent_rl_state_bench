"""Template-derived synthetic task helpers.

These helpers are for high-level categories whose quality mostly comes from
interaction structure rather than a single parameterized policy calculation.
They keep the original environment and expected state assertions intact while
creating new task ids and light surface-language variants.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.paths import PACKAGE_ROOT
from state_bench.schemas import TaskDefinition, UserSimulatorConfig


def official_customer_support_root() -> Path:
    return PACKAGE_ROOT / "domains" / "customer_support"


def load_official_task(task_id: str) -> tuple[TaskDefinition, CSEnvironmentData]:
    root = official_customer_support_root()
    return (
        TaskDefinition.load(root / "tasks" / f"{task_id}.json"),
        CSEnvironmentData.load(root / "task_envs" / f"{task_id}.json"),
    )


def clone_task_from_template(
    *,
    source_task_id: str,
    new_task_id: str,
    task_type: str,
    surface_variant: int,
) -> tuple[TaskDefinition, CSEnvironmentData, dict[str, Any]]:
    source_task, source_env = load_official_task(source_task_id)
    source = source_task.to_dict()
    source["task_id"] = new_task_id
    source["task_type"] = task_type
    source["task_requirements"] = _retag_requirements(
        source.get("task_requirements") or [],
        old_prefix=source_task.task_id,
        new_prefix=new_task_id,
    )
    source["opening_message"] = _surface_opening(source_task.opening_message, surface_variant)
    source["task_summary"] = _surface_summary(source_task.task_summary, task_type, source_task_id, surface_variant)
    source["user_simulator"] = _surface_simulator(source_task.user_simulator, task_type, surface_variant)

    task = TaskDefinition.from_dict(source)
    metadata = {
        "task_id": new_task_id,
        "source_task_id": source_task_id,
        "generator_version": f"{task_type}_template_v0.1",
        "task_family": task_type,
        "mutation_level": "template_surface",
        "difficulty": _difficulty_from_source_id(source_task_id),
        "changed_factors": ["official_template_structure", f"surface_variant_{surface_variant}"],
        "requires_confirmation": _has_state_mutation(task),
        "is_compound": task_type == "compound",
        "gold_tool_plan": None,
        "expected_state_requirements": copy.deepcopy(task.state_requirements),
        "language_rewritten": False,
        "judge": None,
    }
    return task, CSEnvironmentData.from_dict(copy.deepcopy(source_env.to_dict())), metadata


def _retag_requirements(requirements: list[dict[str, Any]], *, old_prefix: str, new_prefix: str) -> list[dict[str, Any]]:
    retagged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, req in enumerate(copy.deepcopy(requirements), 1):
        req_id = str(req.get("id") or f"requirement_{idx}")
        if old_prefix and req_id.startswith(old_prefix):
            req_id = f"{new_prefix}{req_id[len(old_prefix):]}"
        elif req_id[0].isdigit() or req_id.startswith(("hard_", "spare_", "challenge_")):
            req_id = f"{new_prefix}_{idx:02d}_{req_id}"
        else:
            req_id = f"{new_prefix}_{req_id}"
        if req_id in seen:
            req_id = f"{req_id}_{idx:02d}"
        seen.add(req_id)
        req["id"] = req_id
        retagged.append(req)
    return retagged


def _surface_opening(opening: str, variant: int) -> str:
    if variant % 4 == 0:
        return opening
    if variant % 4 == 1:
        return f"Can you help me with this? {opening}"
    if variant % 4 == 2:
        return f"I need this handled carefully: {opening}"
    return f"Please check the policy and do the right thing here. {opening}"


def _surface_summary(summary: str, task_type: str, source_task_id: str, variant: int) -> str:
    prefix = (
        f"Synthetic {task_type} task derived from official customer_support template "
        f"{source_task_id}, surface variant {variant}. "
    )
    return prefix + summary


def _surface_simulator(simulator: UserSimulatorConfig, task_type: str, variant: int) -> dict[str, Any]:
    data = simulator.to_dict()
    data["user_sim_context"] = (
        f"You are in a synthetic {task_type} benchmark scenario. "
        f"Follow the original task facts exactly; do not introduce new order ids, products, prices, or policy facts. "
        f"Surface variant: {variant}.\n\n{data['user_sim_context']}"
    )
    rules = list(data.get("task_rules") or [])
    rules.append("Do not change the intended resolution path from the task summary.")
    data["task_rules"] = rules
    return data


def _difficulty_from_source_id(source_task_id: str) -> str:
    if source_task_id.startswith("hard_") or "-hard_" in source_task_id:
        return "hard"
    if source_task_id.startswith("challenge_") or "-challenge_" in source_task_id:
        return "challenge"
    if source_task_id.startswith("spare_") or "-spare_" in source_task_id:
        return "medium"
    return "medium"


def _has_state_mutation(task: TaskDefinition) -> bool:
    return bool(task.state_requirements)


def load_template_ids(task_type: str) -> list[str]:
    root = official_customer_support_root() / "tasks"
    ids: list[str] = []
    for path in sorted(root.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if data.get("task_type") == task_type:
            ids.append(data["task_id"])
    return ids
