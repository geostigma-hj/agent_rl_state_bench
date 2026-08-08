"""CLI for synthetic customer_support task generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from state_bench.synthetic.customer_support.cancel import generate_cancel_tasks
from state_bench.synthetic.customer_support.checker import check_task_files
from state_bench.synthetic.customer_support.exchange import generate_exchange_tasks
from state_bench.synthetic.customer_support.return_refund import generate_return_refund_tasks
from state_bench.synthetic.customer_support.warranty import generate_warranty_tasks

FAMILIES = ("return_refund", "cancel", "exchange", "warranty")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic customer_support tasks")
    parser.add_argument("--family", choices=[*FAMILIES, "all"], default="return_refund")
    parser.add_argument("--output-root", type=Path, default=Path("synthetic_data/customer_support"))
    parser.add_argument("--seed-task-id", default="7-return_restocking_waived")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rewrite", action="store_true", help="Use DeepSeek V4 Flash to rewrite language fields")
    args = parser.parse_args()

    selected = FAMILIES if args.family == "all" else (args.family,)
    records = []
    for family in selected:
        if family == "return_refund":
            records.extend(
                generate_return_refund_tasks(
                    seed_task_id=args.seed_task_id,
                    output_root=args.output_root,
                    limit=args.limit,
                    rewrite=args.rewrite,
                )
            )
        elif family == "cancel":
            records.extend(generate_cancel_tasks(output_root=args.output_root, limit=args.limit, rewrite=args.rewrite))
        elif family == "exchange":
            records.extend(generate_exchange_tasks(output_root=args.output_root, limit=args.limit, rewrite=args.rewrite))
        elif family == "warranty":
            records.extend(generate_warranty_tasks(output_root=args.output_root, limit=args.limit, rewrite=args.rewrite))

    results = []
    for record in records:
        result = check_task_files(record["task"], record["env"], record["metadata"])
        results.append(
            {
                "task_id": result.task_id,
                "passed": result.passed,
                "errors": result.errors,
                "warnings": result.warnings,
            }
        )
    print(json.dumps({"generated": len(records), "checks": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
