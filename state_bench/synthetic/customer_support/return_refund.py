"""Programmatic return/refund synthetic task generator."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.paths import PACKAGE_ROOT
from state_bench.schemas import TaskDefinition, UserSimulatorConfig
from state_bench.synthetic.customer_support.checker import (
    replay_gold_tool_plan,
    requirements_from_state_diff,
)
from state_bench.synthetic.customer_support.llm import OpenAICompatibleJsonClient

GENERATOR_VERSION = "return_refund_v0.1"


@dataclass(frozen=True)
class ReturnVariant:
    suffix: str
    mutation_level: str
    difficulty: str
    customer_id: str
    membership_tier: str
    has_prime_shipping: bool
    reason: str
    requested_reason_label: str
    delivery_date: str
    order_date: str
    changed_factors: list[str]
    store_credit_pushback: bool = False
    user_mislabels_defect: bool = False


DEFAULT_RETURN_VARIANTS: tuple[ReturnVariant, ...] = (
    ReturnVariant(
        suffix="standard_changed_mind_restock",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_002",
        membership_tier="standard",
        has_prime_shipping=False,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["membership_tier", "restocking_fee"],
    ),
    ReturnVariant(
        suffix="gold_changed_mind_discount",
        mutation_level="constraint",
        difficulty="medium",
        customer_id="cust_003",
        membership_tier="gold",
        has_prime_shipping=True,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["membership_tier", "restocking_discount"],
    ),
    ReturnVariant(
        suffix="platinum_store_credit_grace",
        mutation_level="constraint",
        difficulty="hard",
        customer_id="cust_001",
        membership_tier="platinum",
        has_prime_shipping=True,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-05-01T10:00:00",
        delivery_date="2026-05-03T14:00:00",
        changed_factors=["return_window", "store_credit_only", "membership_tier"],
        store_credit_pushback=True,
    ),
    ReturnVariant(
        suffix="defective_full_refund",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_004",
        membership_tier="silver",
        has_prime_shipping=False,
        reason="defective",
        requested_reason_label="defective",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["return_reason", "restocking_fee_exemption"],
    ),
    ReturnVariant(
        suffix="cosmetic_claim_not_defective",
        mutation_level="constraint",
        difficulty="hard",
        customer_id="cust_005",
        membership_tier="standard",
        has_prime_shipping=False,
        reason="changed_mind",
        requested_reason_label="defective",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["user_mislabel", "return_reason", "restocking_fee"],
        user_mislabels_defect=True,
    ),
    ReturnVariant(
        suffix="silver_changed_mind_restock",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_004",
        membership_tier="silver",
        has_prime_shipping=False,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-06-02T09:00:00",
        delivery_date="2026-06-06T14:00:00",
        changed_factors=["membership_tier", "restocking_fee"],
    ),
    ReturnVariant(
        suffix="platinum_changed_mind_discount",
        mutation_level="constraint",
        difficulty="medium",
        customer_id="cust_001",
        membership_tier="platinum",
        has_prime_shipping=True,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-06-03T10:00:00",
        delivery_date="2026-06-07T15:30:00",
        changed_factors=["membership_tier", "restocking_discount"],
    ),
    ReturnVariant(
        suffix="standard_wrong_item_full_refund",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_002",
        membership_tier="standard",
        has_prime_shipping=False,
        reason="wrong_item",
        requested_reason_label="wrong item",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["return_reason", "merchant_fault"],
    ),
    ReturnVariant(
        suffix="gold_wrong_item_full_refund",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_003",
        membership_tier="gold",
        has_prime_shipping=True,
        reason="wrong_item",
        requested_reason_label="wrong item",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["return_reason", "merchant_fault", "membership_tier"],
    ),
    ReturnVariant(
        suffix="silver_damaged_in_transit_full_refund",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_004",
        membership_tier="silver",
        has_prime_shipping=False,
        reason="damaged_in_transit",
        requested_reason_label="damaged in transit",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["return_reason", "merchant_fault"],
    ),
    ReturnVariant(
        suffix="standard_store_credit_grace",
        mutation_level="constraint",
        difficulty="hard",
        customer_id="cust_002",
        membership_tier="standard",
        has_prime_shipping=False,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-05-18T10:00:00",
        delivery_date="2026-05-20T14:00:00",
        changed_factors=["return_window", "store_credit_only"],
        store_credit_pushback=True,
    ),
    ReturnVariant(
        suffix="gold_store_credit_grace",
        mutation_level="constraint",
        difficulty="hard",
        customer_id="cust_003",
        membership_tier="gold",
        has_prime_shipping=True,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-04-20T10:00:00",
        delivery_date="2026-04-25T14:00:00",
        changed_factors=["return_window", "store_credit_only", "membership_tier"],
        store_credit_pushback=True,
    ),
    ReturnVariant(
        suffix="prime_standard_changed_mind_restock",
        mutation_level="state",
        difficulty="medium",
        customer_id="cust_002",
        membership_tier="standard",
        has_prime_shipping=True,
        reason="changed_mind",
        requested_reason_label="changed mind",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["prime_shipping", "return_window", "restocking_fee"],
    ),
    ReturnVariant(
        suffix="silver_defective_full_refund",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_004",
        membership_tier="silver",
        has_prime_shipping=False,
        reason="defective",
        requested_reason_label="defective",
        order_date="2026-06-03T11:00:00",
        delivery_date="2026-06-06T13:00:00",
        changed_factors=["return_reason", "restocking_fee_exemption"],
    ),
    ReturnVariant(
        suffix="platinum_damaged_full_refund",
        mutation_level="state",
        difficulty="easy",
        customer_id="cust_001",
        membership_tier="platinum",
        has_prime_shipping=True,
        reason="damaged_in_transit",
        requested_reason_label="damaged in transit",
        order_date="2026-06-01T10:00:00",
        delivery_date="2026-06-05T14:00:00",
        changed_factors=["return_reason", "merchant_fault", "membership_tier"],
    ),
)


def generate_return_refund_tasks(
    *,
    seed_task_id: str = "7-return_restocking_waived",
    output_root: Path | None = None,
    limit: int | None = None,
    rewrite: bool = False,
) -> list[dict[str, Path]]:
    repo_root = PACKAGE_ROOT.parent
    output_root = output_root or repo_root / "synthetic_data" / "customer_support"
    tasks_dir = output_root / "tasks"
    envs_dir = output_root / "task_envs"
    metadata_dir = output_root / "metadata"
    for directory in (tasks_dir, envs_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed_task = TaskDefinition.load(repo_root / "state_bench" / "domains" / "customer_support" / "tasks" / f"{seed_task_id}.json")
    seed_env = CSEnvironmentData.load(
        repo_root / "state_bench" / "domains" / "customer_support" / "task_envs" / f"{seed_task_id}.json"
    )

    variants = DEFAULT_RETURN_VARIANTS[:limit] if limit is not None else DEFAULT_RETURN_VARIANTS
    rewriter = OpenAICompatibleJsonClient.deepseek_v4_flash() if rewrite else None
    written: list[dict[str, Path]] = []
    for idx, variant in enumerate(variants, 1):
        task, env_data, metadata = build_return_refund_variant(seed_task, seed_env, variant, idx)
        if rewriter is not None:
            _rewrite_language(task, metadata, rewriter)

        env_path = envs_dir / f"{task.task_id}.json"
        task_path = tasks_dir / f"{task.task_id}.json"
        metadata_path = metadata_dir / f"{task.task_id}.json"
        task.task_env_path = str(Path("synthetic_data") / "customer_support" / "task_envs" / env_path.name)

        env_data.save(env_path)
        _write_json(task_path, task.to_dict())
        _write_json(metadata_path, metadata)
        written.append({"task": task_path, "env": env_path, "metadata": metadata_path})
    return written


def build_return_refund_variant(
    seed_task: TaskDefinition,
    seed_env: CSEnvironmentData,
    variant: ReturnVariant,
    index: int,
) -> tuple[TaskDefinition, CSEnvironmentData, dict[str, Any]]:
    env_dict = copy.deepcopy(seed_env.to_dict())
    env_data = CSEnvironmentData.from_dict(env_dict)
    order = env_data.orders[0]
    item = env_data.order_items[0]
    product = next(p for p in env_data.products if p.product_id == item.product_id)
    customer = next(c for c in env_data.customers if c.customer_id == variant.customer_id)

    task_id = f"syn-return-{index:03d}-{variant.suffix}"
    now = "2026-06-12T10:00:00"

    order.customer_id = customer.customer_id
    order.order_date = variant.order_date
    order.status = "delivered"
    order.shipping_status = "delivered"
    order.delivery_date = variant.delivery_date
    order.shipping_cost = 0
    order.payment_method = "credit_card"
    order.payment_details = {"credit_card": item.unit_price}
    order.subtotal = item.unit_price
    order.discount_code = None
    order.discount_amount = 0
    order.total_paid = item.unit_price
    order.is_gift = False
    order.gift_sender = None

    item.item_status = "delivered"
    item.return_reason = None
    item.refund_amount = None
    item.refund_method = None
    item.restocking_fee = None
    item.return_label_issued = False
    item.replacement_item_id = None
    item.goodwill_credit = 0
    item.goodwill_credit_method = None
    item.store_credit_only = False

    customer.membership_tier = variant.membership_tier
    customer.has_prime_shipping = variant.has_prime_shipping
    customer.preferred_refund_method = "store_credit" if variant.store_credit_pushback else "original_payment"
    product.category = "electronics"
    product.restocking_fee_pct = 15

    gold_tool_plan = _build_gold_return_plan(env_data, now, item.item_id, variant.reason)
    state_diff = replay_gold_tool_plan(env_data, now, gold_tool_plan)
    state_requirements = requirements_from_state_diff(state_diff)

    task = TaskDefinition(
        task_id=task_id,
        task_summary=_task_summary(product.name, order.order_id, variant),
        user_id=customer.customer_id,
        now=now,
        opening_message=_opening_message(product.name, order.order_id, variant),
        user_simulator=UserSimulatorConfig(
            user_sim_context=_user_sim_context(product.name, order.order_id, item.item_id, variant),
            known_info=[
                f"You bought {product.name} in order {order.order_id}.",
                f"You received the order on {order.delivery_date}.",
            ],
            unknown_info=[
                "You do not know the exact return policy calculation.",
                "You do not know the internal item_id unless the agent tells you.",
            ],
            task_rules=_task_rules(variant),
        ),
        task_type="return_item",
        state_requirements=state_requirements,
        task_requirements=_task_requirements(task_id, item.item_id, variant),
        task_env_path=None,
    )
    metadata = {
        "task_id": task_id,
        "source_seed": seed_task.task_id,
        "generator_version": GENERATOR_VERSION,
        "task_family": "return_refund",
        "mutation_level": variant.mutation_level,
        "difficulty": variant.difficulty,
        "changed_factors": variant.changed_factors,
        "requires_confirmation": True,
        "is_compound": False,
        "gold_tool_plan": gold_tool_plan,
        "expected_state_diff": state_diff.to_dict(),
        "language_rewritten": False,
        "judge": None,
    }
    return task, env_data, metadata


def _build_gold_return_plan(env_data: CSEnvironmentData, now: str, item_id: str, reason: str) -> list[dict[str, Any]]:
    env = CustomerSupportEnvironment(env_data.deep_copy(), now)
    env.get_policies({"topic": "return"})
    preview = env.process_return({"item_id": item_id, "reason": reason, "confirm": False})
    if "error" in preview or preview.get("status") == "rejected":
        raise ValueError(f"return preview failed for {item_id}: {preview}")
    amount = int(preview["refund_amount"])
    return [
        {"tool": "get_policies", "params": {"topic": "return"}},
        {"tool": "process_return", "params": {"item_id": item_id, "reason": reason, "confirm": False}},
        {"tool": "process_return", "params": {"item_id": item_id, "reason": reason, "amount": amount, "confirm": True}},
    ]


def _opening_message(product_name: str, order_id: str, variant: ReturnVariant) -> str:
    if variant.store_credit_pushback:
        return (
            f"I want to return the {product_name} from order {order_id}. I know it has been a while, "
            "but I want the money back on my card."
        )
    if variant.user_mislabels_defect:
        return (
            f"I need to return the {product_name} from order {order_id}. It is defective: "
            "the color and look are not what I expected at all."
        )
    if variant.reason == "defective":
        return f"The {product_name} from order {order_id} is defective and I need to return it to my original payment method."
    return (
        f"I would like to return the {product_name} from order {order_id}. "
        "I opened it, but I do not want it anymore, and I want the refund back to my original payment method."
    )


def _task_summary(product_name: str, order_id: str, variant: ReturnVariant) -> str:
    if variant.user_mislabels_defect:
        return (
            f"Customer describes {product_name} from {order_id} as defective, but the described issue is cosmetic "
            "preference. The correct resolution is a changed-mind return with normal electronics restocking policy."
        )
    if variant.store_credit_pushback:
        return (
            f"Customer wants to return {product_name} from {order_id} outside the full-refund window but inside "
            "the store-credit grace period. The return is allowed, but the refund method must remain store credit."
        )
    return (
        f"Customer wants to return {product_name} from {order_id}. The agent must apply the return policy, preview "
        "the return, obtain confirmation, and process the correct refund."
    )


def _user_sim_context(product_name: str, order_id: str, item_id: str, variant: ReturnVariant) -> str:
    context = [
        f"You want help returning {product_name} from order {order_id}.",
        f"The relevant item is {item_id}.",
        f"Your stated reason is {variant.requested_reason_label}.",
        f"Your preferred refund method is {'store credit' if variant.store_credit_pushback else 'original payment'}.",
        "If the agent asks for confirmation after explaining the refund, confirm that you want to proceed.",
    ]
    if variant.store_credit_pushback:
        context.append("If the agent says the refund must be store credit, push once for a card refund, then accept the policy.")
    if variant.user_mislabels_defect:
        context.append("If asked what is defective, explain that the product works but the color/appearance is disappointing.")
    return " ".join(context)


def _task_rules(variant: ReturnVariant) -> list[str]:
    rules = [
        "Do not invent order IDs, item IDs, prices, dates, or policy details.",
        "If the agent previews a valid return and asks for confirmation, agree to proceed.",
    ]
    if variant.store_credit_pushback:
        rules.append("Push back once for original-payment refund, but accept store credit if the agent explains the policy.")
    if variant.user_mislabels_defect:
        rules.append("When asked to clarify the defect, say the item functions correctly but does not match your aesthetic expectations.")
    return rules


def _task_requirements(task_id: str, item_id: str, variant: ReturnVariant) -> list[dict[str, Any]]:
    requirements = [
        {
            "id": f"{task_id}_must_review_return_policy",
            "kind": "must",
            "requirement": "Agent must review the return policy before using process_return.",
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_preview_before_confirm",
            "kind": "must",
            "requirement": (
                f"Agent must make a process_return preview tool call for {item_id} with confirm false or omitted "
                "before making the confirm=true process_return tool call."
            ),
            "evidence": "tool_calls",
        },
        {
            "id": f"{task_id}_must_process_correct_reason",
            "kind": "must",
            "requirement": f"Agent must process the item with return reason {variant.reason}.",
            "evidence": "tool_calls",
        },
    ]
    if variant.store_credit_pushback:
        requirements.append(
            {
                "id": f"{task_id}_must_not_flip_store_credit",
                "kind": "must_not",
                "requirement": "Agent must not change the refund method from store credit to original payment.",
                "evidence": "conversation_or_tool_calls",
            }
        )
    if variant.user_mislabels_defect:
        requirements.append(
            {
                "id": f"{task_id}_must_not_accept_false_defect",
                "kind": "must_not",
                "requirement": "Agent must not process or describe the return as a defective-item exception.",
                "evidence": "conversation_or_tool_calls",
            }
        )
    return requirements


def _rewrite_language(task: TaskDefinition, metadata: dict[str, Any], client: OpenAICompatibleJsonClient) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
