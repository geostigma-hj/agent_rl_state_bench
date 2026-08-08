import json
from pathlib import Path

from state_bench.synthetic.customer_support.cancel import generate_cancel_tasks
from state_bench.synthetic.customer_support.checker import check_task_files
from state_bench.synthetic.customer_support.exchange import generate_exchange_tasks
from state_bench.synthetic.customer_support.price_match import generate_price_match_tasks
from state_bench.synthetic.customer_support.return_refund import generate_return_refund_tasks
from state_bench.synthetic.customer_support.shipping import generate_shipping_tasks
from state_bench.synthetic.customer_support.warranty import generate_warranty_tasks


def test_return_refund_generator_writes_checkable_task(tmp_path: Path) -> None:
    records = generate_return_refund_tasks(output_root=tmp_path / "synthetic", limit=1, rewrite=False)

    assert len(records) == 1
    record = records[0]
    result = check_task_files(record["task"], record["env"], record["metadata"])

    assert result.passed
    task = json.loads(record["task"].read_text())
    metadata = json.loads(record["metadata"].read_text())
    assert task["task_type"] == "return_item"
    assert task["state_requirements"]
    assert metadata["gold_tool_plan"]
    assert "api_key" not in json.dumps(metadata).lower()


def test_return_refund_generator_can_emit_multiple_mutations(tmp_path: Path) -> None:
    records = generate_return_refund_tasks(output_root=tmp_path / "synthetic", limit=3, rewrite=False)
    task_ids = [json.loads(record["task"].read_text())["task_id"] for record in records]
    difficulties = [json.loads(record["metadata"].read_text())["difficulty"] for record in records]

    assert len(set(task_ids)) == 3
    assert set(difficulties) >= {"easy", "medium"}


def test_core_family_generators_emit_checkable_tasks(tmp_path: Path) -> None:
    generators = {
        "cancel": generate_cancel_tasks,
        "exchange": generate_exchange_tasks,
        "price_match": generate_price_match_tasks,
        "shipping": generate_shipping_tasks,
        "warranty": generate_warranty_tasks,
    }
    for family, generator in generators.items():
        records = generator(output_root=tmp_path / family, limit=2, rewrite=False)
        assert len(records) == 2
        for record in records:
            result = check_task_files(record["task"], record["env"], record["metadata"])
            metadata = json.loads(record["metadata"].read_text())
            assert result.passed, result.errors
            assert metadata["task_family"] == family
            assert metadata["gold_tool_plan"]
            assert metadata["expected_state_diff"]
