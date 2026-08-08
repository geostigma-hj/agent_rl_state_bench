"""Programmatic shipping-damage synthetic task generator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.schemas import TaskDefinition, UserSimulatorConfig
from state_bench.synthetic.customer_support.core import build_state_requirements, finalize_task, load_seed

GENERATOR_VERSION = "shipping_v0.1"


@dataclass(frozen=True)
class ShippingDamageVariant:
    suffix: str
    difficulty: str
    customer_id: str
    membership_tier: str
    product_name: str
    item_price: int
    is_fragile: bool
    requested_resolution: str
    changed_factors: list[str]


BASE_SHIPPING_VARIANTS: tuple[ShippingDamageVariant, ...] = (
    ShippingDamageVariant("fragile_vase_refund_goodwill", "medium", "cust_003", "gold", "Artisan Crystal Vase", 109, True, "refund", ["fragile_goodwill"]),
    ShippingDamageVariant("fragile_lamp_refund_goodwill", "medium", "cust_004", "silver", "Ceramic Table Lamp", 129, True, "refund", ["fragile_goodwill", "membership_tier"]),
    ShippingDamageVariant("fragile_glassware_refund_goodwill", "medium", "cust_002", "standard", "Crystal Glassware Set", 89, True, "refund", ["fragile_goodwill"]),
    ShippingDamageVariant("nonfragile_headphones_refund", "easy", "cust_002", "standard", "SoundMax Basic Headphones", 149, False, "refund", ["damaged_no_goodwill"]),
    ShippingDamageVariant("nonfragile_shoes_refund", "easy", "cust_005", "standard", "TrailStep Running Shoes", 129, False, "refund", ["damaged_no_goodwill"]),
    ShippingDamageVariant("fragile_blender_refund_goodwill", "medium", "cust_001", "platinum", "PowerBlend Glass Blender", 159, True, "refund", ["fragile_goodwill", "membership_tier"]),
    ShippingDamageVariant("nonfragile_adapter_refund", "easy", "cust_004", "silver", "DeskHub USB-C Adapter", 45, False, "refund", ["low_value_item"]),
    ShippingDamageVariant("fragile_espresso_part_goodwill", "hard", "cust_003", "gold", "BrewMaster Glass Carafe", 79, True, "refund", ["fragile_goodwill", "small_item"]),
    ShippingDamageVariant("nonfragile_tablet_refund", "medium", "cust_001", "platinum", "SlateTab Pro 11-inch", 599, False, "refund", ["high_value_item"]),
    ShippingDamageVariant("fragile_frame_refund_goodwill", "medium", "cust_005", "standard", "Framed Wall Art", 119, True, "refund", ["fragile_goodwill"]),
)


def _auto_shipping_variants() -> tuple[ShippingDamageVariant, ...]:
    products = [
        ("Porcelain Dinner Set", 149, True),
        ("Glass Coffee Carafe", 69, True),
        ("SmartHome Display", 199, False),
        ("Ceramic Serving Bowl", 79, True),
        ("NoiseBlock Earbuds", 119, False),
        ("Travel Backpack", 89, False),
        ("Framed Wall Mirror", 189, True),
        ("Kitchen Knife Set", 139, False),
        ("Crystal Wine Decanter", 159, True),
        ("Laptop Sleeve", 49, False),
    ]
    customers = [
        ("cust_001", "platinum"),
        ("cust_002", "standard"),
        ("cust_003", "gold"),
        ("cust_004", "silver"),
        ("cust_005", "standard"),
    ]
    variants: list[ShippingDamageVariant] = []
    seen = {variant.suffix for variant in BASE_SHIPPING_VARIANTS}
    idx = 0
    for product_name, price, fragile in products:
        for customer_id, tier in customers:
            suffix = f"auto_{idx+1:03d}_{tier}_{'fragile' if fragile else 'nonfragile'}_{product_name.lower().replace(' ', '_').replace('-', '_')}"
            if suffix in seen:
                idx += 1
                continue
            seen.add(suffix)
            factors = ["fragile_goodwill"] if fragile else ["damaged_no_goodwill"]
            if price >= 500:
                factors.append("high_value_item")
            if price < 75:
                factors.append("low_value_item")
            variants.append(
                ShippingDamageVariant(
                    suffix,
                    "medium" if fragile else "easy",
                    customer_id,
                    tier,
                    product_name,
                    price,
                    fragile,
                    "refund",
                    factors,
                )
            )
            idx += 1
            if len(BASE_SHIPPING_VARIANTS) + len(variants) >= 50:
                return tuple(variants)
    return tuple(variants)


DEFAULT_SHIPPING_VARIANTS: tuple[ShippingDamageVariant, ...] = (*BASE_SHIPPING_VARIANTS, *_auto_shipping_variants())


def generate_shipping_tasks(*, output_root: Path, limit: int | None = None, rewrite: bool = False) -> list[dict[str, Path]]:
    seed_task, seed_env = load_seed("21-shipping_damaged_fragile")
    variants = DEFAULT_SHIPPING_VARIANTS[:limit] if limit is not None else DEFAULT_SHIPPING_VARIANTS
    records = []
    for idx, variant in enumerate(variants, 1):
        task, env_data, metadata = build_shipping_damage_variant(seed_task, seed_env, variant, idx)
        records.append(finalize_task(task=task, env_data=env_data, metadata=metadata, output_root=output_root, rewrite=rewrite))
    return records


def build_shipping_damage_variant(
    seed_task: TaskDefinition,
    seed_env: CSEnvironmentData,
    variant: ShippingDamageVariant,
    index: int,
) -> tuple[TaskDefinition, CSEnvironmentData, dict[str, Any]]:
    env_data = CSEnvironmentData.from_dict(copy.deepcopy(seed_env.to_dict()))
    order = env_data.orders[0]
    item = env_data.order_items[0]
    product = env_data.products[0]
    customer = next(c for c in env_data.customers if c.customer_id == variant.customer_id)
    task_id = f"syn-shipping-{index:03d}-{variant.suffix}"
    now = "2026-06-12T10:00:00"

    order.customer_id = customer.customer_id
    order.status = "delivered"
    order.shipping_status = "damaged"
    order.delivery_date = "2026-06-08T14:00:00"
    order.delivery_promised_date = "2026-06-05T18:00:00"
    order.shipping_cost = 0
    order.payment_method = "credit_card"
    order.payment_details = {"credit_card": variant.item_price}
    order.subtotal = variant.item_price
    order.total_paid = variant.item_price
    order.discount_code = None
    order.discount_amount = 0

    customer.membership_tier = variant.membership_tier
    product.name = variant.product_name
    product.price = variant.item_price
    product.current_price = None
    product.is_fragile = variant.is_fragile
    product.restocking_fee_pct = 0
    item.unit_price = variant.item_price
    item.item_status = "delivered"
    item.return_reason = None
    item.refund_amount = None
    item.refund_method = None
    item.restocking_fee = None
    item.return_label_issued = False
    item.goodwill_credit = 0
    item.goodwill_credit_method = None

    gold_tool_plan = _build_gold_shipping_damage_plan(env_data, item.item_id, variant.item_price, variant.is_fragile)
    state_requirements, state_diff = build_state_requirements(env_data, now, gold_tool_plan)
    task = TaskDefinition(
        task_id=task_id,
        task_summary=(
            f"Customer reports {product.name} from {order.order_id} arrived damaged in transit. "
            "Correct handling is the shipping-damage path: review shipping and return policy, refund the item, "
            "issue a return label, and include fragile goodwill only when applicable."
        ),
        user_id=customer.customer_id,
        now=now,
        opening_message=f"My {product.name} from order {order.order_id} arrived damaged. The package was crushed.",
        user_simulator=UserSimulatorConfig(
            user_sim_context=(
                f"You received {product.name} from order {order.order_id} damaged in transit. "
                "You want a refund to original payment. If the item is fragile, ask whether a separate fragile-item goodwill credit applies."
            ),
            known_info=[f"You bought {product.name} in order {order.order_id}.", "The item arrived damaged with crushed packaging."],
            unknown_info=["You do not know the internal item_id unless the agent tells you."],
            task_rules=[
                "Keep the issue scoped to shipping damage; do not ask for a changed-mind return, cancellation, warranty claim, or exchange.",
                "Confirm after a valid damaged-in-transit return preview. If the item is fragile, expect the separate goodwill credit.",
            ],
        ),
        task_type="shipping_claim",
        state_requirements=state_requirements,
        task_requirements=_requirements(task_id, item.item_id, variant.is_fragile),
    )
    metadata = {
        "task_id": task_id,
        "source_seed": seed_task.task_id,
        "generator_version": GENERATOR_VERSION,
        "task_family": "shipping",
        "mutation_level": "state",
        "difficulty": variant.difficulty,
        "changed_factors": variant.changed_factors,
        "requires_confirmation": True,
        "is_compound": variant.is_fragile,
        "gold_tool_plan": gold_tool_plan,
        "expected_state_diff": state_diff,
        "language_rewritten": False,
        "judge": None,
    }
    return task, env_data, metadata


def _build_gold_shipping_damage_plan(env_data: CSEnvironmentData, item_id: str, amount: int, is_fragile: bool) -> list[dict[str, Any]]:
    env = CustomerSupportEnvironment(env_data.deep_copy(), "2026-06-12T10:00:00")
    env.get_policies({"topic": "shipping"})
    env.get_policies({"topic": "return"})
    preview_params = {"item_id": item_id, "reason": "damaged_in_transit", "confirm": False}
    preview = env.process_return(preview_params)
    if "error" in preview or preview.get("status") == "rejected":
        raise ValueError(f"shipping damage return preview failed: {preview}")
    confirm_params = {**preview_params, "confirm": True, "amount": preview["refund_amount"]}
    plan = [
        {"tool": "get_policies", "params": {"topic": "shipping"}},
        {"tool": "get_policies", "params": {"topic": "return"}},
        {"tool": "process_return", "params": preview_params},
        {"tool": "process_return", "params": confirm_params},
    ]
    if is_fragile:
        env.process_return(confirm_params)
        env.get_policies({"topic": "refund"})
        refund_params = {"item_id": item_id, "amount": 10, "refund_method": "original_payment", "confirm": False}
        refund_preview = env.process_refund(refund_params)
        if "error" in refund_preview:
            raise ValueError(f"shipping goodwill preview failed: {refund_preview}")
        plan.extend(
            [
                {"tool": "get_policies", "params": {"topic": "refund"}},
                {"tool": "process_refund", "params": refund_params},
                {"tool": "process_refund", "params": {**refund_params, "confirm": True}},
            ]
        )
    return plan


def _requirements(task_id: str, item_id: str, is_fragile: bool) -> list[dict[str, Any]]:
    requirements = [
        {
            "id": f"{task_id}_must_review_shipping_policy",
            "kind": "must",
            "requirement": "Agent must review the shipping policy before resolving the damaged-in-transit claim.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_use_damaged_reason",
            "kind": "must",
            "requirement": f"Agent must process item {item_id} with reason damaged_in_transit, not changed_mind.",
            "evidence": "tool_calls",
        },
    ]
    if is_fragile:
        requirements.append(
            {
                "id": f"{task_id}_must_issue_fragile_goodwill",
                "kind": "must",
                "requirement": "Agent must issue the separate $10 fragile-item goodwill credit to original payment.",
                "evidence": "tool_calls",
            }
        )
    else:
        requirements.append(
            {
                "id": f"{task_id}_must_not_issue_fragile_goodwill",
                "kind": "must_not",
                "requirement": "Agent must not issue a fragile-item goodwill credit for a non-fragile product.",
                "evidence": "tool_calls",
            }
        )
    return requirements
