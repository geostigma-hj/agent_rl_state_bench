"""Deterministic validation for synthetic customer_support tasks."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.schemas import StateDiff, TaskDefinition
from state_bench.scoring import evaluate_state_requirements


@dataclass
class CheckResult:
    task_id: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    state_diff: dict[str, Any] | None = None


def check_task_files(task_path: Path, env_path: Path, metadata_path: Path | None = None) -> CheckResult:
    task = TaskDefinition.load(task_path)
    env_data = CSEnvironmentData.load(env_path)
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_check_referential_integrity(env_data))
    errors.extend(_check_task_references(task, env_data))

    state_diff_dict = None
    if metadata_path and metadata_path.exists():
        metadata = _load_json(metadata_path)
        plan = metadata.get("gold_tool_plan")
        if plan:
            try:
                state_diff = replay_gold_tool_plan(env_data, task.now, plan)
                state_diff_dict = state_diff.to_dict()
                state_score = evaluate_state_requirements(task, state_diff)
                if state_score is None or state_score.score != 1:
                    errors.append(f"state_requirements do not match replayed gold_tool_plan: {state_score}")
            except Exception as exc:
                errors.append(f"gold_tool_plan replay failed: {exc}")
        else:
            warnings.append("metadata has no gold_tool_plan")

    return CheckResult(
        task_id=task.task_id,
        passed=not errors,
        errors=errors,
        warnings=warnings,
        state_diff=state_diff_dict,
    )


def replay_gold_tool_plan(env_data: CSEnvironmentData, now: str, plan: list[dict[str, Any]]) -> StateDiff:
    env = CustomerSupportEnvironment(env_data.deep_copy(), now)
    before = copy.deepcopy(env.get_full_snapshot())
    for step in plan:
        tool = step.get("tool")
        params = step.get("params", {})
        handler = env.tool_handlers.get(tool)
        if handler is None:
            raise ValueError(f"unknown tool in gold_tool_plan: {tool!r}")
        result = handler(params)
        if isinstance(result, dict) and "error" in result:
            raise ValueError(f"{tool} returned error: {result['error']}")
        if isinstance(result, dict) and result.get("status") == "rejected":
            raise ValueError(f"{tool} rejected plan step: {result}")
    after = env.get_full_snapshot()
    return StateDiff.compute(before, after)


def requirements_from_state_diff(state_diff: StateDiff) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for entity_type, records in sorted(state_diff.modified.items()):
        for record_key, changed_fields in sorted(records.items()):
            for field_name, change in sorted(changed_fields.items()):
                requirements.append(
                    {
                        "entity_type": entity_type,
                        "record_key": record_key,
                        "field": field_name,
                        "expected_value": change["new"],
                    }
                )
    for entity_type, records in sorted(state_diff.created.items()):
        for _record_key, record in sorted(records.items()):
            stable_match = _created_record_match_fields(record)
            expected_fields = {k: v for k, v in record.items() if k not in stable_match}
            requirements.append(
                {
                    "entity_type": entity_type,
                    "match_fields": stable_match,
                    "expected_fields": expected_fields,
                }
            )
    if state_diff.deleted:
        raise ValueError("deleted records are not supported in synthetic state requirements")
    return requirements


def _created_record_match_fields(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("item_id", "warranty_id", "order_id", "customer_id", "product_id"):
        if key in record:
            return {key: record[key]}
    raise ValueError(f"created record lacks a stable id field: {record}")


def _check_referential_integrity(env_data: CSEnvironmentData) -> list[str]:
    errors: list[str] = []
    customers = {c.customer_id for c in env_data.customers}
    orders = {o.order_id for o in env_data.orders}
    products = {p.product_id for p in env_data.products}
    items = {i.item_id for i in env_data.order_items}

    for order in env_data.orders:
        if order.customer_id not in customers:
            errors.append(f"order {order.order_id} references missing customer {order.customer_id}")
    for item in env_data.order_items:
        if item.order_id not in orders:
            errors.append(f"item {item.item_id} references missing order {item.order_id}")
        if item.product_id not in products:
            errors.append(f"item {item.item_id} references missing product {item.product_id}")
    for warranty in env_data.warranties:
        if warranty.order_id not in orders:
            errors.append(f"warranty {warranty.warranty_id} references missing order {warranty.order_id}")
        if warranty.item_id not in items:
            errors.append(f"warranty {warranty.warranty_id} references missing item {warranty.item_id}")
        if warranty.product_id not in products:
            errors.append(f"warranty {warranty.warranty_id} references missing product {warranty.product_id}")
    return errors


def _check_task_references(task: TaskDefinition, env_data: CSEnvironmentData) -> list[str]:
    errors: list[str] = []
    customers = {c.customer_id for c in env_data.customers}
    if task.user_id not in customers:
        errors.append(f"task user_id {task.user_id} is not present in env")

    snapshot = {
        "orders": {o.order_id: o.to_dict() for o in env_data.orders},
        "order_items": {i.item_id: i.to_dict() for i in env_data.order_items},
        "customers": {c.customer_id: c.to_dict() for c in env_data.customers},
        "warranties": {w.warranty_id: w.to_dict() for w in env_data.warranties},
    }
    for req in task.state_requirements:
        entity_type = req.get("entity_type")
        record_key = req.get("record_key")
        if record_key and entity_type in snapshot and record_key not in snapshot[entity_type]:
            # Created records use match_fields instead of record_key, so a direct
            # record_key should usually exist in the initial env.
            errors.append(f"state requirement references missing {entity_type}.{record_key}")
    return errors


def _load_json(path: Path) -> dict[str, Any]:
    import json

    with open(path) as f:
        return json.load(f)
