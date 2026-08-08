"""LLM judging for synthetic customer_support tasks."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from state_bench.domains.customer_support.environment import CustomerSupportEnvironment
from state_bench.domains.customer_support.schemas import CSEnvironmentData
from state_bench.synthetic.customer_support.checker import check_task_files
from state_bench.synthetic.customer_support.llm import OpenAICompatibleJsonClient

VALID_DECISIONS = {"ACCEPT", "REVIEW", "REJECT"}


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / requests_per_minute
        self._lock = Lock()
        self._next_request_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_request_at:
                time.sleep(self._next_request_at - now)
                now = time.monotonic()
            self._next_request_at = now + self._interval


def judge_task(
    *,
    task_path: Path,
    env_path: Path,
    metadata_path: Path,
    output_path: Path,
    judge_provider: str = "step",
    client: OpenAICompatibleJsonClient | None = None,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    client = client or _build_judge_client(judge_provider)
    task = _load_json(task_path)
    env = _load_json(env_path)
    metadata = _load_json(metadata_path)
    deterministic = check_task_files(task_path, env_path, metadata_path)

    try:
        if rate_limiter is not None:
            rate_limiter.acquire()
        judgment = client.complete_json(
            system=_judge_system_prompt(),
            user=json.dumps(
                {
                    "task": task,
                    "environment": _compact_env(env),
                    "metadata": _compact_metadata(metadata),
                    "gold_tool_replay_trace": _gold_tool_replay_trace(env, task["now"], metadata.get("gold_tool_plan") or []),
                    "deterministic_check": {
                        "passed": deterministic.passed,
                        "errors": deterministic.errors,
                        "warnings": deterministic.warnings,
                    },
                    "required_output_schema": {
                        "decision": "ACCEPT | REVIEW | REJECT",
                        "confidence": "number from 0 to 1",
                        "issues": ["list of concrete issues, empty if none"],
                        "ambiguities": ["list of ambiguity risks, empty if none"],
                        "recommendation": "short actionable recommendation",
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
    except Exception as exc:
        judgment = {
            "decision": "REVIEW",
            "confidence": None,
            "issues": [f"Judge call or JSON parsing failed: {type(exc).__name__}: {str(exc)[:300]}"],
            "ambiguities": [],
            "recommendation": "Manual adjudication required because the judge did not return valid JSON.",
        }
    decision = str(judgment.get("decision", "REVIEW")).upper()
    judgment.setdefault("issues", [])
    judgment.setdefault("ambiguities", [])
    judgment.setdefault("recommendation", "")
    judgment.setdefault("confidence", None)
    if decision not in VALID_DECISIONS:
        decision = "REVIEW"
        judgment["issues"] = [*judgment.get("issues", []), "Judge returned invalid decision; coerced to REVIEW."]
    if not deterministic.passed and decision == "ACCEPT":
        decision = "REVIEW"
        judgment["issues"] = [
            *judgment.get("issues", []),
            "Deterministic checks failed; cannot auto-accept even if LLM judge approved.",
        ]
    if deterministic.passed and decision == "REVIEW" and not judgment["issues"] and not judgment["ambiguities"]:
        decision = "ACCEPT"
        judgment["recommendation"] = "Auto-accepted: deterministic checks passed and judge reported no concrete issues."
    judgment["decision"] = decision
    judgment["judge_model"] = client.model
    judgment["deterministic_check_passed"] = deterministic.passed
    judgment["manual_adjudication"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, judgment)
    metadata["judge"] = {
        "decision": judgment["decision"],
        "judge_model": judgment["judge_model"],
        "deterministic_check_passed": judgment["deterministic_check_passed"],
        "issues": judgment.get("issues", []),
        "ambiguities": judgment.get("ambiguities", []),
        "manual_adjudication": None,
    }
    _write_json(metadata_path, metadata)
    return judgment


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge synthetic customer_support tasks with an OpenAI-compatible LLM")
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic_data/customer_support"))
    parser.add_argument("--task-id", action="append", default=None, help="Task id to judge. May be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--judge-provider",
        choices=("step", "deepseek"),
        default=os.environ.get("SYNTH_JUDGE_PROVIDER", "step"),
        help="LLM provider for synthetic judging (default: env SYNTH_JUDGE_PROVIDER or step).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SYNTH_STEP_JUDGE_WORKERS", "5")),
        help="Number of concurrent Step judge workers (default: 5, matching the observed StepFun concurrency cap).",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=int(os.environ.get("SYNTH_STEP_JUDGE_RPM_LIMIT", "9")),
        help="Maximum Step judge requests per minute. Use 0 to disable throttling (default: 9).",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.rpm_limit < 0:
        parser.error("--rpm-limit must be >= 0")

    task_ids = args.task_id or sorted(path.stem for path in (args.synthetic_root / "tasks").glob("*.json"))
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    results: list[dict[str, Any]] = []
    rate_limiter = RateLimiter(args.rpm_limit) if args.rpm_limit else None
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                judge_task,
                task_path=args.synthetic_root / "tasks" / f"{task_id}.json",
                env_path=args.synthetic_root / "task_envs" / f"{task_id}.json",
                metadata_path=args.synthetic_root / "metadata" / f"{task_id}.json",
                output_path=args.synthetic_root / "judgments" / f"{task_id}.json",
                judge_provider=args.judge_provider,
                client=None,
                rate_limiter=rate_limiter,
            ): task_id
            for task_id in task_ids
        }
        for future in as_completed(futures):
            task_id = futures[future]
            judgment = future.result()
            item = {"task_id": task_id, "decision": judgment["decision"], "confidence": judgment.get("confidence")}
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    print(json.dumps({"judged": len(results), "results": results}, indent=2, ensure_ascii=False))


def _build_judge_client(judge_provider: str) -> OpenAICompatibleJsonClient:
    if judge_provider == "step":
        return OpenAICompatibleJsonClient.step37_flash()
    if judge_provider == "deepseek":
        return OpenAICompatibleJsonClient.deepseek_v4_flash()
    raise ValueError(f"Unsupported judge provider: {judge_provider}")


def _judge_system_prompt() -> str:
    return (
        "You are an independent synthetic benchmark data auditor for STATE-Bench Customer Support. "
        "Judge whether the task is clear, executable, internally consistent, and has a single reasonable final state. "
        "Check consistency among opening_message, user_simulator, environment, task_requirements, state_requirements, "
        "and gold_tool_plan. Do not solve with hidden assumptions. Prefer REVIEW over ACCEPT when there is ambiguity. "
        "For split-payment refunds, distinguish API response allocation from persisted item state: process_refund may "
        "return refund_method='split' with split_details while order_items.refund_method remains the requested "
        "high-level method such as 'original_payment'; state_requirements evaluate persisted state, not response-only fields. "
        "Return exactly one JSON object with keys: decision, confidence, issues, ambiguities, recommendation. "
        "decision must be ACCEPT, REVIEW, or REJECT. issues and ambiguities must be arrays of strings. "
        "If deterministic checks passed and you find no concrete issue or ambiguity, return decision=ACCEPT."
    )


def _compact_env(env: dict[str, Any]) -> dict[str, Any]:
    return {
        "products": env.get("products", []),
        "orders": env.get("orders", []),
        "order_items": env.get("order_items", []),
        "customers": env.get("customers", []),
        "warranties": env.get("warranties", []),
    }


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "task_id",
        "source_seed",
        "generator_version",
        "task_family",
        "mutation_level",
        "difficulty",
        "changed_factors",
        "requires_confirmation",
        "is_compound",
        "gold_tool_plan",
        "expected_state_diff",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def _gold_tool_replay_trace(env: dict[str, Any], now: str, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    env_data = CSEnvironmentData.from_dict(env)
    environment = CustomerSupportEnvironment(env_data, now)
    trace = []
    for step in plan:
        tool = step.get("tool")
        params = step.get("params", {})
        handler = environment.tool_handlers.get(tool)
        if handler is None:
            trace.append({"tool": tool, "params": params, "result": {"error": f"unknown tool {tool!r}"}})
            break
        result = handler(params)
        trace.append({"tool": tool, "params": params, "result": result})
        if isinstance(result, dict) and ("error" in result or result.get("status") == "rejected"):
            break
    return trace


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
