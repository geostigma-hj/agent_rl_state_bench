"""Programmatic warranty synthetic task generator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.schemas import TaskDefinition, UserSimulatorConfig
from state_bench.synthetic.customer_support.core import build_state_requirements, finalize_task, load_seed

GENERATOR_VERSION = "warranty_v0.1"


@dataclass(frozen=True)
class WarrantyVariant:
    suffix: str
    difficulty: str
    item_price: int
    product_name: str
    warranty_end: str
    claim_count: int
    max_claims: int
    changed_factors: list[str]


DEFAULT_WARRANTY_VARIANTS: tuple[WarrantyVariant, ...] = (
    WarrantyVariant("active_repair", "easy", 249, "SoundMax Wireless Headphones", "2026-07-31", 0, 3, ["active_warranty", "repair"]),
    WarrantyVariant("active_low_value_replacement", "easy", 89, "PowerBlend Mini Blender", "2026-12-31", 0, 3, ["low_value", "replacement"]),
    WarrantyVariant("recurring_replacement", "medium", 249, "SoundMax Wireless Headphones", "2026-07-31", 2, 3, ["recurring_defect"]),
    WarrantyVariant("recent_expiry_discounted_repair", "medium", 249, "SoundMax Wireless Headphones", "2026-06-01", 0, 3, ["recent_expiry"]),
    WarrantyVariant("maxed_paid_repair", "hard", 599, "SlateTab Pro 11-inch", "2027-07-01", 3, 3, ["claim_limit", "paid_repair"]),
    WarrantyVariant("active_tablet_repair", "easy", 599, "SlateTab Pro 11-inch", "2026-12-31", 0, 3, ["active_warranty", "repair"]),
    WarrantyVariant("active_low_value_usb_replacement", "easy", 45, "DeskHub USB-C Adapter", "2026-12-31", 0, 3, ["low_value", "replacement"]),
    WarrantyVariant("recurring_low_value_replacement", "medium", 79, "PowerBlend Mini Blender", "2026-11-30", 2, 3, ["recurring_defect", "low_value"]),
    WarrantyVariant("recent_expiry_low_value_discounted_repair", "medium", 89, "PowerBlend Mini Blender", "2026-06-01", 0, 3, ["recent_expiry", "discounted_repair"]),
    WarrantyVariant("recent_expiry_tablet_discounted_repair", "medium", 599, "SlateTab Pro 11-inch", "2026-05-25", 1, 3, ["recent_expiry", "discounted_repair"]),
    WarrantyVariant("long_expiry_paid_options", "hard", 249, "SoundMax Wireless Headphones", "2026-04-01", 0, 3, ["expired_warranty", "paid_repair_options"]),
    WarrantyVariant("maxed_low_value_paid_repair", "hard", 89, "PowerBlend Mini Blender", "2026-12-31", 3, 3, ["claim_limit", "paid_repair"]),
    WarrantyVariant("maxed_headphones_paid_repair", "hard", 249, "SoundMax Wireless Headphones", "2027-01-31", 3, 3, ["claim_limit", "paid_repair"]),
)


def generate_warranty_tasks(*, output_root: Path, limit: int | None = None, rewrite: bool = False) -> list[dict[str, Path]]:
    seed_task, seed_env = load_seed("30-warranty_manufacturer")
    variants = DEFAULT_WARRANTY_VARIANTS[:limit] if limit is not None else DEFAULT_WARRANTY_VARIANTS
    records = []
    for idx, variant in enumerate(variants, 1):
        task, env_data, metadata = build_warranty_variant(seed_task, seed_env, variant, idx)
        records.append(finalize_task(task=task, env_data=env_data, metadata=metadata, output_root=output_root, rewrite=rewrite))
    return records


def build_warranty_variant(
    seed_task: TaskDefinition,
    seed_env: CSEnvironmentData,
    variant: WarrantyVariant,
    index: int,
) -> tuple[TaskDefinition, CSEnvironmentData, dict[str, Any]]:
    env_data = CSEnvironmentData.from_dict(copy.deepcopy(seed_env.to_dict()))
    order = env_data.orders[0]
    item = env_data.order_items[0]
    product = env_data.products[0]
    warranty = env_data.warranties[0]
    task_id = f"syn-warranty-{index:03d}-{variant.suffix}"
    now = "2026-06-12T10:00:00"

    product.name = variant.product_name
    product.price = variant.item_price
    product.category = "electronics" if variant.item_price >= 100 else "kitchen"
    product.warranty_months = 12
    item.unit_price = variant.item_price
    item.item_status = "delivered"
    item.refund_amount = None
    item.refund_method = None
    item.replacement_item_id = None
    order.subtotal = variant.item_price
    order.total_paid = variant.item_price
    order.payment_details = {"credit_card": variant.item_price}
    order.status = "delivered"
    order.shipping_status = "delivered"
    order.delivery_date = "2025-08-05T14:00:00"
    warranty.end_date = variant.warranty_end
    warranty.claim_count = variant.claim_count
    warranty.max_claims = variant.max_claims
    warranty.status = "active"
    warranty.resolution = None

    gold_tool_plan = _build_gold_warranty_plan(env_data, now, warranty.warranty_id, item.item_id)
    state_requirements, state_diff = build_state_requirements(env_data, now, gold_tool_plan)
    task = TaskDefinition(
        task_id=task_id,
        task_summary=(
            f"Customer asks for warranty service on {product.name} from {order.order_id}. "
            "Correct handling is to check warranty status, review warranty policy, preview, confirm, and file the policy-correct claim."
        ),
        user_id=order.customer_id,
        now=now,
        opening_message=f"My {product.name} from order {order.order_id} is malfunctioning. Can I use the warranty?",
        user_simulator=UserSimulatorConfig(
            user_sim_context=(
                f"You want warranty help for {product.name} from order {order.order_id}. "
                "If the agent explains the warranty outcome and asks for confirmation, agree to proceed."
            ),
            known_info=[f"You bought {product.name} in order {order.order_id}.", "The product is malfunctioning."],
            unknown_info=["You do not know the claim count or exact warranty resolution."],
            task_rules=["Confirm after a valid warranty preview. Do not ask for a return unless the agent says warranty is unavailable."],
        ),
        task_type="warranty_claim",
        state_requirements=state_requirements,
        task_requirements=_requirements(task_id, warranty.warranty_id, item.item_id),
    )
    metadata = {
        "task_id": task_id,
        "source_seed": seed_task.task_id,
        "generator_version": GENERATOR_VERSION,
        "task_family": "warranty",
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


def _build_gold_warranty_plan(env_data: CSEnvironmentData, now: str, warranty_id: str, item_id: str) -> list[dict[str, Any]]:
    env = CustomerSupportEnvironment(env_data.deep_copy(), now)
    env.get_warranty_status({"item_id": item_id})
    env.get_policies({"topic": "warranty"})
    params = {"warranty_id": warranty_id, "item_id": item_id, "issue_description": "Product is malfunctioning", "confirm": False}
    preview = env.process_warranty_claim(params)
    if "error" in preview or preview.get("status") == "rejected":
        raise ValueError(f"warranty preview failed: {preview}")
    return [
        {"tool": "get_warranty_status", "params": {"item_id": item_id}},
        {"tool": "get_policies", "params": {"topic": "warranty"}},
        {"tool": "process_warranty_claim", "params": params},
        {"tool": "process_warranty_claim", "params": {**params, "confirm": True}},
    ]


def _requirements(task_id: str, warranty_id: str, item_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{task_id}_must_check_warranty_status",
            "kind": "must",
            "requirement": f"Agent must check warranty status for {item_id}.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_review_warranty_policy",
            "kind": "must",
            "requirement": "Agent must review warranty policy before filing a warranty claim.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_preview_before_confirm",
            "kind": "must",
            "requirement": (
                f"Agent must make a process_warranty_claim preview tool call for {warranty_id} with confirm false "
                "or omitted before making the confirm=true process_warranty_claim tool call."
            ),
            "evidence": "tool_calls",
        },
    ]
