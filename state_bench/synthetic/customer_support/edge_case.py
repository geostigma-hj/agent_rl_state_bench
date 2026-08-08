"""Template-derived edge-case synthetic task generator."""

from __future__ import annotations

from pathlib import Path

from state_bench.synthetic.customer_support.core import finalize_task
from state_bench.synthetic.customer_support.template_clone import clone_task_from_template, load_template_ids

GENERATOR_VERSION = "edge_case_template_v0.1"
DEFAULT_EDGE_CASE_COUNT = 20


def generate_edge_case_tasks(*, output_root: Path, limit: int | None = None, rewrite: bool = False) -> list[dict[str, Path]]:
    source_ids = load_template_ids("edge_case")
    count = limit if limit is not None else DEFAULT_EDGE_CASE_COUNT
    records: list[dict[str, Path]] = []
    for idx in range(1, count + 1):
        source_task_id = source_ids[(idx - 1) % len(source_ids)]
        cycle = (idx - 1) // len(source_ids)
        task_id = f"syn-edge-{idx:03d}-{_slug(source_task_id)}"
        task, env_data, metadata = clone_task_from_template(
            source_task_id=source_task_id,
            new_task_id=task_id,
            task_type="edge_case",
            surface_variant=cycle,
        )
        metadata["generator_version"] = GENERATOR_VERSION
        records.append(finalize_task(task=task, env_data=env_data, metadata=metadata, output_root=output_root, rewrite=rewrite))
    return records


def _slug(task_id: str) -> str:
    return task_id.split("-", 1)[1] if "-" in task_id else task_id
