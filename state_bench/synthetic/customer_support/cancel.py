"""Programmatic cancellation synthetic task generator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.schemas import TaskDefinition, UserSimulatorConfig
from state_bench.synthetic.customer_support.core import build_state_requirements, finalize_task, load_seed

GENERATOR_VERSION = "cancel_v0.1"


@dataclass(frozen=True)
class CancelVariant:
    suffix: str
    difficulty: str
    order_status: str
    shipping_status: str
    item_statuses: tuple[str, ...]
    target_item_indices: tuple[int, ...] | None
    payment_method: str
    changed_factors: list[str]


DEFAULT_CANCEL_VARIANTS: tuple[CancelVariant, ...] = (
    CancelVariant("pre_shipment_full", "easy", "processing", "processing", ("confirmed", "confirmed"), None, "credit_card", ["shipping_status"]),
    CancelVariant("in_transit_full_intercept_fee", "medium", "shipped", "in_transit", ("shipped", "shipped"), None, "credit_card", ["intercept_fee"]),
    CancelVariant("partial_pre_shipment_item", "medium", "processing", "processing", ("shipped", "confirmed"), (1,), "credit_card", ["partial_cancel"]),
    CancelVariant("split_payment_in_transit", "hard", "shipped", "in_transit", ("shipped", "shipped"), None, "split", ["split_payment", "intercept_fee"]),
    CancelVariant("lost_package_free_cancel", "medium", "shipped", "lost", ("shipped", "shipped"), None, "credit_card", ["lost_shipping_status"]),
    CancelVariant("debit_pre_shipment_full", "easy", "processing", "processing", ("confirmed", "confirmed"), None, "debit_card", ["payment_method"]),
    CancelVariant("gift_card_pre_shipment_full", "easy", "processing", "processing", ("confirmed", "confirmed"), None, "gift_card", ["payment_method"]),
    CancelVariant("paypal_pre_shipment_full", "easy", "processing", "processing", ("confirmed", "confirmed"), None, "paypal", ["payment_method"]),
    CancelVariant("partial_processing_first_item", "medium", "processing", "processing", ("confirmed", "confirmed"), (0,), "credit_card", ["partial_cancel"]),
    CancelVariant("partial_in_transit_first_item_fee", "hard", "shipped", "in_transit", ("shipped", "shipped"), (0,), "credit_card", ["partial_cancel", "intercept_fee"]),
    CancelVariant("partial_in_transit_second_item_fee", "hard", "shipped", "in_transit", ("shipped", "shipped"), (1,), "credit_card", ["partial_cancel", "intercept_fee"]),
    CancelVariant("split_payment_lost_full", "medium", "shipped", "lost", ("shipped", "shipped"), None, "split", ["split_payment", "lost_shipping_status"]),
    CancelVariant("debit_in_transit_full_fee", "medium", "shipped", "in_transit", ("shipped", "shipped"), None, "debit_card", ["payment_method", "intercept_fee"]),
    CancelVariant("partial_lost_first_item_free", "medium", "shipped", "lost", ("shipped", "shipped"), (0,), "credit_card", ["partial_cancel", "lost_shipping_status"]),
    CancelVariant("partial_lost_second_item_free", "medium", "shipped", "lost", ("shipped", "shipped"), (1,), "credit_card", ["partial_cancel", "lost_shipping_status"]),
)


def generate_cancel_tasks(*, output_root: Path, limit: int | None = None, rewrite: bool = False) -> list[dict[str, Path]]:
    seed_task, seed_env = load_seed("12-cancel_after_shipment")
    variants = DEFAULT_CANCEL_VARIANTS[:limit] if limit is not None else DEFAULT_CANCEL_VARIANTS
    records = []
    for idx, variant in enumerate(variants, 1):
        task, env_data, metadata = build_cancel_variant(seed_task, seed_env, variant, idx)
        records.append(finalize_task(task=task, env_data=env_data, metadata=metadata, output_root=output_root, rewrite=rewrite))
    return records


def build_cancel_variant(
    seed_task: TaskDefinition,
    seed_env: CSEnvironmentData,
    variant: CancelVariant,
    index: int,
) -> tuple[TaskDefinition, CSEnvironmentData, dict[str, Any]]:
    env_data = CSEnvironmentData.from_dict(copy.deepcopy(seed_env.to_dict()))
    order = env_data.orders[0]
    items = env_data.order_items
    task_id = f"syn-cancel-{index:03d}-{variant.suffix}"
    now = "2026-06-12T10:00:00"

    order.status = variant.order_status
    order.shipping_status = variant.shipping_status
    order.delivery_date = None
    order.payment_method = variant.payment_method
    order.subtotal = sum(item.unit_price for item in items)
    order.total_paid = order.subtotal
    if variant.payment_method == "split":
        order.payment_details = {"credit_card": items[0].unit_price, "gift_card": items[1].unit_price}
    else:
        order.payment_details = {variant.payment_method: order.total_paid}
    for item, status in zip(items, variant.item_statuses, strict=True):
        item.item_status = status
        item.refund_amount = None
        item.refund_method = None

    target_ids = [items[i].item_id for i in variant.target_item_indices] if variant.target_item_indices is not None else None
    gold_tool_plan = _build_gold_cancel_plan(env_data, now, order.order_id, target_ids)
    state_requirements, state_diff = build_state_requirements(env_data, now, gold_tool_plan)
    product_names = [next(p.name for p in env_data.products if p.product_id == item.product_id) for item in items]

    task = TaskDefinition(
        task_id=task_id,
        task_summary=_summary(order.order_id, product_names, variant),
        user_id=order.customer_id,
        now=now,
        opening_message=_opening(order.order_id, product_names, variant),
        user_simulator=UserSimulatorConfig(
            user_sim_context=_sim_context(order.order_id, product_names, target_ids, variant),
            known_info=[f"You placed order {order.order_id}.", f"The order contains: {', '.join(product_names)}."],
            unknown_info=["You do not know whether a cancellation fee applies."],
            task_rules=[
                "If the agent previews a valid cancellation and asks for confirmation, agree to proceed.",
                "Do not ask to cancel items outside the stated scope.",
            ],
        ),
        task_type="cancel_order",
        state_requirements=state_requirements,
        task_requirements=_requirements(task_id, order.order_id, target_ids),
    )
    metadata = {
        "task_id": task_id,
        "source_seed": seed_task.task_id,
        "generator_version": GENERATOR_VERSION,
        "task_family": "cancel",
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


def _build_gold_cancel_plan(env_data: CSEnvironmentData, now: str, order_id: str, item_ids: list[str] | None) -> list[dict[str, Any]]:
    env = CustomerSupportEnvironment(env_data.deep_copy(), now)
    env.get_policies({"topic": "cancellation"})
    params = {"order_id": order_id, "confirm": False}
    if item_ids is not None:
        params["item_ids"] = item_ids
    preview = env.cancel_order(params)
    if "error" in preview or preview.get("status") == "rejected":
        raise ValueError(f"cancel preview failed: {preview}")
    confirm_params = {**params, "confirm": True}
    return [
        {"tool": "get_policies", "params": {"topic": "cancellation"}},
        {"tool": "cancel_order", "params": params},
        {"tool": "cancel_order", "params": confirm_params},
    ]


def _opening(order_id: str, product_names: list[str], variant: CancelVariant) -> str:
    if variant.target_item_indices is not None:
        names = [product_names[i] for i in variant.target_item_indices]
        return f"I need to cancel just the {', '.join(names)} from order {order_id}."
    return f"I need to cancel order {order_id}. I do not need the {', '.join(product_names)} anymore."


def _summary(order_id: str, product_names: list[str], variant: CancelVariant) -> str:
    scope = "selected item(s)" if variant.target_item_indices is not None else "the full order"
    return f"Customer wants to cancel {scope} in {order_id}. Correct handling is to apply cancellation policy, preview, confirm, and update only the eligible items."


def _sim_context(order_id: str, product_names: list[str], target_ids: list[str] | None, variant: CancelVariant) -> str:
    scope = f"only item(s) {target_ids}" if target_ids else "the full order"
    return (
        f"You want to cancel {scope} from order {order_id}, which contains {', '.join(product_names)}. "
        "If the agent explains a valid cancellation preview, confirm that you want to proceed."
    )


def _requirements(task_id: str, order_id: str, target_ids: list[str] | None) -> list[dict[str, Any]]:
    scope = f"items {target_ids}" if target_ids else f"order {order_id}"
    return [
        {
            "id": f"{task_id}_must_review_cancellation_policy",
            "kind": "must",
            "requirement": "Agent must review the cancellation policy before using cancel_order.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_preview_before_confirm",
            "kind": "must",
            "requirement": (
                f"Agent must make a cancel_order preview tool call for {scope} with confirm false or omitted "
                "before making the confirm=true cancel_order tool call."
            ),
            "evidence": "tool_calls",
        },
    ]
