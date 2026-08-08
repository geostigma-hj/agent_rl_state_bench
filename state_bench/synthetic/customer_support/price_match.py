"""Programmatic price-match synthetic task generator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.schemas import TaskDefinition, UserSimulatorConfig
from state_bench.synthetic.customer_support.core import build_state_requirements, finalize_task, load_seed

GENERATOR_VERSION = "price_match_v0.1"


@dataclass(frozen=True)
class PriceMatchVariant:
    suffix: str
    difficulty: str
    customer_id: str
    membership_tier: str
    product_name: str
    original_price: int
    current_price: int
    quoted_drop: int | None
    payment_method: str
    changed_factors: list[str]


BASE_PRICE_MATCH_VARIANTS: tuple[PriceMatchVariant, ...] = (
    PriceMatchVariant("standard_exact_drop", "easy", "cust_002", "standard", "SoundMax Basic Headphones", 149, 139, 10, "credit_card", ["price_drop"]),
    PriceMatchVariant("customer_overquotes_drop", "medium", "cust_002", "standard", "SoundMax Basic Headphones", 149, 139, 30, "credit_card", ["customer_wrong_amount"]),
    PriceMatchVariant("gold_small_drop", "easy", "cust_003", "gold", "PowerBlend Mini Blender", 89, 79, 10, "credit_card", ["membership_tier"]),
    PriceMatchVariant("platinum_large_drop", "medium", "cust_001", "platinum", "SlateTab Pro 11-inch", 599, 549, 50, "credit_card", ["large_price_drop"]),
    PriceMatchVariant("split_tender_adjustment", "hard", "cust_001", "platinum", "TechPhone Pro 16", 799, 749, 50, "split", ["split_payment"]),
    PriceMatchVariant("gift_card_adjustment", "medium", "cust_004", "silver", "Artisan Crystal Vase", 109, 99, 10, "gift_card", ["payment_method"]),
    PriceMatchVariant("paypal_adjustment", "medium", "cust_005", "standard", "ComfyCotton Shirt", 59, 49, 10, "paypal", ["payment_method"]),
    PriceMatchVariant("customer_underquotes_drop", "medium", "cust_003", "gold", "SoundMax Wireless Headphones", 249, 219, 20, "credit_card", ["customer_wrong_amount"]),
    PriceMatchVariant("small_five_dollar_drop", "easy", "cust_004", "silver", "DeskHub USB-C Adapter", 45, 40, 5, "credit_card", ["small_price_drop"]),
    PriceMatchVariant("large_laptop_drop", "medium", "cust_003", "gold", "ProBook Laptop", 799, 729, 70, "credit_card", ["high_value_item"]),
    PriceMatchVariant("no_customer_amount_given", "easy", "cust_002", "standard", "PowerBlend Mini Blender", 89, 84, None, "credit_card", ["agent_must_compute"]),
    PriceMatchVariant("split_tender_customer_overquotes", "hard", "cust_001", "platinum", "TechPhone Pro 16", 799, 769, 80, "split", ["split_payment", "customer_wrong_amount"]),
    PriceMatchVariant("standard_twenty_five_drop", "medium", "cust_005", "standard", "TrailStep Running Shoes", 129, 104, 25, "debit_card", ["payment_method"]),
    PriceMatchVariant("gold_current_price_lookup", "medium", "cust_003", "gold", "BrewMaster Espresso Machine", 349, 319, None, "credit_card", ["agent_must_lookup"]),
    PriceMatchVariant("silver_fifteen_drop", "easy", "cust_004", "silver", "NoiseBlock Earbuds", 119, 104, 15, "credit_card", ["price_drop"]),
)


def _auto_price_match_variants() -> tuple[PriceMatchVariant, ...]:
    products = [
        ("NoiseBlock Earbuds", 119),
        ("PowerBlend Mini Blender", 89),
        ("SoundMax Wireless Headphones", 249),
        ("SlateTab Pro 11-inch", 599),
        ("TrailStep Running Shoes", 129),
        ("DeskHub USB-C Adapter", 45),
        ("BrewMaster Espresso Machine", 349),
        ("ProBook Laptop", 799),
        ("ComfyCotton Shirt", 59),
        ("AirPure Compact Purifier", 159),
        ("FlexCharge Travel Dock", 69),
        ("ViewPlus 27-inch Monitor", 499),
        ("SoundMax Studio Headphones", 299),
        ("ChefCore Stand Mixer", 229),
        ("UrbanLite Carry-On Suitcase", 179),
    ]
    customers = [
        ("cust_001", "platinum"),
        ("cust_002", "standard"),
        ("cust_003", "gold"),
        ("cust_004", "silver"),
        ("cust_005", "standard"),
    ]
    payments = ["credit_card", "debit_card", "paypal", "gift_card", "split"]
    drops = [5, 10, 15, 20, 25, 30, 40, 50]
    variants: list[PriceMatchVariant] = []
    seen = {variant.suffix for variant in BASE_PRICE_MATCH_VARIANTS}
    idx = 0
    for product_name, original_price in products:
        for customer_id, tier in customers:
            drop = drops[idx % len(drops)]
            if drop >= original_price:
                continue
            payment = payments[idx % len(payments)]
            quoted = None
            factors = ["price_drop"]
            difficulty = "easy"
            if idx % 3 == 1:
                quoted = drop + 15
                factors.append("customer_wrong_amount")
                difficulty = "medium"
            elif idx % 3 == 2:
                quoted = max(1, drop - 5)
                factors.append("customer_wrong_amount")
                difficulty = "medium"
            if payment == "split":
                factors.append("split_payment")
                difficulty = "hard"
            suffix = f"auto_{idx+1:03d}_{tier}_{payment}_{product_name.lower().replace(' ', '_').replace('-', '_')}_{drop}_drop"
            if suffix in seen:
                idx += 1
                continue
            seen.add(suffix)
            variants.append(
                PriceMatchVariant(
                    suffix,
                    difficulty,
                    customer_id,
                    tier,
                    product_name,
                    original_price,
                    original_price - drop,
                    quoted,
                    payment,
                    factors,
                )
            )
            idx += 1
            if len(BASE_PRICE_MATCH_VARIANTS) + len(variants) >= 90:
                return tuple(variants)
    return tuple(variants)


DEFAULT_PRICE_MATCH_VARIANTS: tuple[PriceMatchVariant, ...] = (*BASE_PRICE_MATCH_VARIANTS, *_auto_price_match_variants())


def generate_price_match_tasks(*, output_root: Path, limit: int | None = None, rewrite: bool = False) -> list[dict[str, Path]]:
    seed_task, seed_env = load_seed("108-hard_price_match_customer_quotes_wrong_amount")
    variants = DEFAULT_PRICE_MATCH_VARIANTS[:limit] if limit is not None else DEFAULT_PRICE_MATCH_VARIANTS
    records = []
    for idx, variant in enumerate(variants, 1):
        task, env_data, metadata = build_price_match_variant(seed_task, seed_env, variant, idx)
        records.append(finalize_task(task=task, env_data=env_data, metadata=metadata, output_root=output_root, rewrite=rewrite))
    return records


def build_price_match_variant(
    seed_task: TaskDefinition,
    seed_env: CSEnvironmentData,
    variant: PriceMatchVariant,
    index: int,
) -> tuple[TaskDefinition, CSEnvironmentData, dict[str, Any]]:
    env_data = CSEnvironmentData.from_dict(copy.deepcopy(seed_env.to_dict()))
    order = env_data.orders[0]
    item = env_data.order_items[0]
    product = env_data.products[0]
    customer = next(c for c in env_data.customers if c.customer_id == variant.customer_id)
    task_id = f"syn-price-{index:03d}-{variant.suffix}"
    now = "2026-07-08T10:00:00"
    refund_amount = variant.original_price - variant.current_price

    order.customer_id = customer.customer_id
    order.order_date = "2026-07-01T10:00:00"
    order.status = "delivered"
    order.shipping_status = "delivered"
    order.delivery_date = "2026-07-05T12:00:00"
    order.delivery_promised_date = "2026-07-04T18:00:00"
    order.payment_method = variant.payment_method
    order.subtotal = variant.original_price
    order.discount_code = None
    order.discount_amount = 0
    order.total_paid = variant.original_price
    order.shipping_cost = 0
    if variant.payment_method == "split":
        order.payment_details = {"credit_card": int(variant.original_price * 0.75), "gift_card": variant.original_price - int(variant.original_price * 0.75)}
    else:
        order.payment_details = {variant.payment_method: variant.original_price}

    customer.membership_tier = variant.membership_tier
    product.name = variant.product_name
    product.price = variant.original_price
    product.current_price = variant.current_price
    product.in_stock = True
    item.unit_price = variant.original_price
    item.item_status = "delivered"
    item.refund_amount = None
    item.refund_method = None
    item.return_reason = None
    item.return_label_issued = False

    gold_tool_plan = _build_gold_price_match_plan(env_data, item.item_id, product.product_id, refund_amount)
    state_requirements, state_diff = build_state_requirements(env_data, now, gold_tool_plan)
    task = TaskDefinition(
        task_id=task_id,
        task_summary=(
            f"Customer requests a price-match adjustment for {product.name} in {order.order_id}. "
            f"The correct outcome is a ${refund_amount} price-match refund while keeping the order and item active."
        ),
        user_id=customer.customer_id,
        now=now,
        opening_message=_opening(product.name, order.order_id, variant, refund_amount),
        user_simulator=UserSimulatorConfig(
            user_sim_context=_sim_context(product.name, order.order_id, variant, refund_amount),
            known_info=_known_info(product.name, order.order_id, variant, refund_amount),
            unknown_info=["You do not know the exact internal price-match workflow.", "You do not know payment allocation details unless the agent explains them."],
            task_rules=[
                "Keep the request scoped to a price-match refund; do not ask to return, cancel, or exchange the item.",
                "If the agent previews the policy-supported amount and keeps the order active, confirm the refund.",
            ],
        ),
        task_type="price_match_refund",
        state_requirements=state_requirements,
        task_requirements=_requirements(task_id, product.product_id, refund_amount),
    )
    metadata = {
        "task_id": task_id,
        "source_seed": seed_task.task_id,
        "generator_version": GENERATOR_VERSION,
        "task_family": "price_match",
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


def _build_gold_price_match_plan(env_data: CSEnvironmentData, item_id: str, product_id: str, amount: int) -> list[dict[str, Any]]:
    env = CustomerSupportEnvironment(env_data.deep_copy(), "2026-07-08T10:00:00")
    env.get_policies({"topic": "refund"})
    env.get_product_details({"product_id": product_id})
    params = {"item_id": item_id, "amount": amount, "refund_method": "original_payment", "confirm": False}
    preview = env.process_refund(params)
    if "error" in preview:
        raise ValueError(f"price-match refund preview failed: {preview}")
    return [
        {"tool": "get_policies", "params": {"topic": "refund"}},
        {"tool": "get_product_details", "params": {"product_id": product_id}},
        {"tool": "process_refund", "params": params},
        {"tool": "process_refund", "params": {**params, "confirm": True}},
    ]


def _opening(product_name: str, order_id: str, variant: PriceMatchVariant, refund_amount: int) -> str:
    if variant.quoted_drop is None:
        return f"The {product_name} from order {order_id} is cheaper now. Can you check the price adjustment?"
    return f"The {product_name} from order {order_id} is cheaper now, so I think I should get ${variant.quoted_drop} back."


def _sim_context(product_name: str, order_id: str, variant: PriceMatchVariant, refund_amount: int) -> str:
    quoted = f"You think the drop is ${variant.quoted_drop}, but accept correction to ${refund_amount} if the agent verifies the product record. " if variant.quoted_drop and variant.quoted_drop != refund_amount else ""
    return (
        f"You want a price-match refund for {product_name} in order {order_id}. "
        f"{quoted}Do not allow a return, cancellation, exchange, or replacement; the item should stay active."
    )


def _known_info(product_name: str, order_id: str, variant: PriceMatchVariant, refund_amount: int) -> list[str]:
    facts = [f"You bought {product_name} in order {order_id}."]
    if variant.quoted_drop is not None and variant.quoted_drop != refund_amount:
        facts.append(f"You believe the price dropped by ${variant.quoted_drop}, but you are not certain.")
    else:
        facts.append(f"The current product price is ${variant.current_price}.")
    return facts


def _requirements(task_id: str, product_id: str, refund_amount: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{task_id}_must_lookup_product_price",
            "kind": "must",
            "requirement": f"Agent must verify current price with get_product_details for {product_id}.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_preview_before_confirm",
            "kind": "must",
            "requirement": f"Agent must preview the ${refund_amount} price-match refund before confirming it.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_not_return_cancel_exchange",
            "kind": "must_not",
            "requirement": "Agent must not return, cancel, or exchange the item.",
            "evidence": "tool_calls",
        },
    ]
