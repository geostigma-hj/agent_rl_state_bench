"""Programmatic exchange synthetic task generator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.schemas import TaskDefinition, UserSimulatorConfig
from state_bench.synthetic.customer_support.core import build_state_requirements, finalize_task, load_seed

GENERATOR_VERSION = "exchange_v0.1"


@dataclass(frozen=True)
class ExchangeVariant:
    suffix: str
    difficulty: str
    new_price: int
    in_stock: bool
    changed_factors: list[str]


DEFAULT_EXCHANGE_VARIANTS: tuple[ExchangeVariant, ...] = (
    ExchangeVariant("same_price_size_swap", "easy", 59, True, ["same_price"]),
    ExchangeVariant("cheaper_store_credit", "medium", 39, True, ["cheaper_item", "store_credit_refund"]),
    ExchangeVariant("more_expensive_customer_pays", "medium", 89, True, ["more_expensive_item"]),
    ExchangeVariant("out_of_stock_store_credit", "hard", 59, False, ["out_of_stock", "store_credit_fallback"]),
    ExchangeVariant("slightly_cheaper_store_credit", "medium", 49, True, ["cheaper_item", "store_credit_refund"]),
    ExchangeVariant("deeply_cheaper_store_credit", "medium", 25, True, ["cheaper_item", "store_credit_refund"]),
    ExchangeVariant("slightly_more_expensive_pays", "medium", 69, True, ["more_expensive_item"]),
    ExchangeVariant("premium_upgrade_pays", "hard", 119, True, ["more_expensive_item", "large_price_difference"]),
    ExchangeVariant("budget_out_of_stock_store_credit", "hard", 39, False, ["out_of_stock", "cheaper_item", "store_credit_fallback"]),
    ExchangeVariant("premium_out_of_stock_store_credit", "hard", 99, False, ["out_of_stock", "more_expensive_item", "store_credit_fallback"]),
    ExchangeVariant("near_same_price_credit", "medium", 58, True, ["cheaper_item", "small_price_difference"]),
    ExchangeVariant("near_same_price_payment", "medium", 60, True, ["more_expensive_item", "small_price_difference"]),
)


def generate_exchange_tasks(*, output_root: Path, limit: int | None = None, rewrite: bool = False) -> list[dict[str, Path]]:
    seed_task, seed_env = load_seed("33-exchange_size_swap")
    variants = DEFAULT_EXCHANGE_VARIANTS[:limit] if limit is not None else DEFAULT_EXCHANGE_VARIANTS
    records = []
    for idx, variant in enumerate(variants, 1):
        task, env_data, metadata = build_exchange_variant(seed_task, seed_env, variant, idx)
        records.append(finalize_task(task=task, env_data=env_data, metadata=metadata, output_root=output_root, rewrite=rewrite))
    return records


def build_exchange_variant(
    seed_task: TaskDefinition,
    seed_env: CSEnvironmentData,
    variant: ExchangeVariant,
    index: int,
) -> tuple[TaskDefinition, CSEnvironmentData, dict[str, Any]]:
    env_data = CSEnvironmentData.from_dict(copy.deepcopy(seed_env.to_dict()))
    order = env_data.orders[0]
    item = env_data.order_items[0]
    old_product = next(p for p in env_data.products if p.product_id == item.product_id)
    new_product = next(p for p in env_data.products if p.product_id != item.product_id)
    task_id = f"syn-exchange-{index:03d}-{variant.suffix}"
    now = "2026-06-12T10:00:00"

    order.status = "delivered"
    order.shipping_status = "delivered"
    order.delivery_date = "2026-06-05T14:00:00"
    item.item_status = "delivered"
    item.refund_amount = None
    item.refund_method = None
    item.replacement_item_id = None
    new_product.price = variant.new_price
    new_product.current_price = None
    new_product.in_stock = variant.in_stock

    gold_tool_plan = _build_gold_exchange_plan(env_data, now, item.item_id, new_product.product_id)
    state_requirements, state_diff = build_state_requirements(env_data, now, gold_tool_plan)
    task = TaskDefinition(
        task_id=task_id,
        task_summary=(
            f"Customer wants to exchange {old_product.name} from {order.order_id} for {new_product.name}. "
            "Correct handling is to apply exchange policy, preview, confirm, and use the policy-defined payment or store-credit outcome."
        ),
        user_id=order.customer_id,
        now=now,
        opening_message=f"I need to exchange the {old_product.name} from order {order.order_id} for {new_product.name}.",
        user_simulator=UserSimulatorConfig(
            user_sim_context=(
                f"You want to exchange {old_product.name} from order {order.order_id} for {new_product.name}. "
                "If the agent previews a valid exchange or store-credit fallback and asks for confirmation, agree to proceed."
            ),
            known_info=[f"You bought {old_product.name} in order {order.order_id}.", f"You want {new_product.name} instead."],
            unknown_info=["You do not know the exchange price-difference policy."],
            task_rules=["Do not ask for a different product. Confirm after a valid preview."],
        ),
        task_type="exchange_item",
        state_requirements=state_requirements,
        task_requirements=_requirements(task_id, item.item_id, new_product.product_id),
    )
    metadata = {
        "task_id": task_id,
        "source_seed": seed_task.task_id,
        "generator_version": GENERATOR_VERSION,
        "task_family": "exchange",
        "mutation_level": "state",
        "difficulty": variant.difficulty,
        "changed_factors": variant.changed_factors,
        "requires_confirmation": True,
        "is_compound": False,
        "gold_tool_plan": gold_tool_plan,
        "expected_state_diff": state_diff,
        "language_rewritten": False,
        "judge": None,
    }
    return task, env_data, metadata


def _build_gold_exchange_plan(env_data: CSEnvironmentData, now: str, item_id: str, new_product_id: str) -> list[dict[str, Any]]:
    env = CustomerSupportEnvironment(env_data.deep_copy(), now)
    env.get_policies({"topic": "exchange"})
    params = {"item_id": item_id, "new_product_id": new_product_id, "confirm": False}
    preview = env.process_exchange(params)
    if "error" in preview or preview.get("status") == "rejected":
        raise ValueError(f"exchange preview failed: {preview}")
    return [
        {"tool": "get_policies", "params": {"topic": "exchange"}},
        {"tool": "process_exchange", "params": params},
        {"tool": "process_exchange", "params": {**params, "confirm": True}},
    ]


def _requirements(task_id: str, item_id: str, new_product_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{task_id}_must_review_exchange_policy",
            "kind": "must",
            "requirement": "Agent must review the exchange policy before using process_exchange.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_preview_before_confirm",
            "kind": "must",
            "requirement": (
                f"Agent must make a process_exchange preview tool call for {item_id} to {new_product_id} "
                "with confirm false or omitted before making the confirm=true process_exchange tool call."
            ),
            "evidence": "tool_calls",
        },
    ]
