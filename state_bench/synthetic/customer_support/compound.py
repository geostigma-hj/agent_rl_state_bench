"""Template-derived compound synthetic task generator."""

from __future__ import annotations

from pathlib import Path

from state_bench.synthetic.customer_support.core import finalize_task
from state_bench.synthetic.customer_support.template_clone import clone_task_from_template, load_template_ids

GENERATOR_VERSION = "compound_template_v0.1"
DEFAULT_COMPOUND_COUNT = 40
BLOCKED_SOURCE_TASK_IDS = {
    "100-challenge_warranty_maxed_return_option",
    "13-cancel_named_then_ambiguous_accessory",
    "41-hard_two_lost_shipments_threshold_split",
    "42-compound_exchange_plus_credit",
    "44-compound_return_gift_exchange",
    "45-compound_partial_cancel_return",
    "52-challenge_fragile_goodwill_separate",
    "62-challenge_exchange_oos_pivot",
    "71-hard_exchange_late_compensation_destination_choice",
    "86-hard_four_path_compound_no_shirt_mutation",
    "93-challenge_seasonal_repeat_shipping",
    "97-challenge_triple_same_category_return",
    "98-challenge_triple_action",
    "140-hard_compound_return_one_price_match_other",
    "142-hard_compound_exchange_plus_late_compensation",
    "143-hard_compound_missing_item_plus_signature_denial",
    "144-hard_compound_gift_return_plus_exchange",
    "145-spare_compound_denial_plus_valid_action",
    "146-hard_compound_wrong_item_plus_late_comp",
    "147-spare_compound_warranty_then_price_match",
}


def generate_compound_tasks(*, output_root: Path, limit: int | None = None, rewrite: bool = False) -> list[dict[str, Path]]:
    source_ids = [task_id for task_id in load_template_ids("compound") if task_id not in BLOCKED_SOURCE_TASK_IDS]
    count = limit if limit is not None else DEFAULT_COMPOUND_COUNT
    records: list[dict[str, Path]] = []
    for idx in range(1, count + 1):
        source_task_id = source_ids[(idx - 1) % len(source_ids)]
        cycle = (idx - 1) // len(source_ids)
        task_id = f"syn-compound-{idx:03d}-{_slug(source_task_id)}"
        task, env_data, metadata = clone_task_from_template(
            source_task_id=source_task_id,
            new_task_id=task_id,
            task_type="compound",
            surface_variant=cycle,
        )
        metadata["generator_version"] = GENERATOR_VERSION
        records.append(finalize_task(task=task, env_data=env_data, metadata=metadata, output_root=output_root, rewrite=rewrite))
    return records


def _slug(task_id: str) -> str:
    return task_id.split("-", 1)[1] if "-" in task_id else task_id
