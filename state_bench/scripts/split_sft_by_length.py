"""Split verl multiturn SFT parquet files by tokenized sequence length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _to_builtin(value.tolist())
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    return value


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def _clean_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        cleaned = []
        for call in tool_calls:
            if not isinstance(call, dict):
                cleaned.append(call)
                continue
            clean_call = dict(call)
            if "arguments" in clean_call:
                clean_call["arguments"] = _drop_none_values(clean_call["arguments"])
            cleaned.append(clean_call)
        message["tool_calls"] = cleaned
    return messages


def _as_list(value: Any) -> list[Any]:
    value = _to_builtin(value)
    return list(value)


def sequence_length(tokenizer: Any, row: pd.Series) -> int:
    messages = _clean_tool_call_arguments(_as_list(row["messages"]))
    tools = _as_list(row["tools"]) if "tools" in row and row["tools"] is not None else None
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=bool(row.get("enable_thinking", False)),
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_one(
    *,
    tokenizer: Any,
    input_path: Path,
    keep_path: Path,
    overflow_path: Path,
    max_length: int,
) -> dict[str, Any]:
    df = pd.read_parquet(input_path)
    lengths = [sequence_length(tokenizer, row) for _, row in df.iterrows()]
    work = df.copy()
    work["sequence_length"] = lengths
    keep = work[work["sequence_length"] <= max_length].copy()
    overflow = work[work["sequence_length"] > max_length].copy()

    keep.to_parquet(keep_path, index=False)
    overflow.to_parquet(overflow_path, index=False)
    return {
        "input": len(work),
        "kept": len(keep),
        "overflow": len(overflow),
        "max_input_length": int(max(lengths) if lengths else 0),
        "overflow_items": overflow[["task_id", "trajectory_path", "sequence_length"]].to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--main-output-dir", type=Path, required=True)
    parser.add_argument("--overflow-output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    args.main_output_dir.mkdir(parents=True, exist_ok=True)
    args.overflow_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    manifest = {"max_length": args.max_length, "splits": {}}
    for split in ["train", "val"]:
        manifest["splits"][split] = split_one(
            tokenizer=tokenizer,
            input_path=args.input_dir / f"{split}.parquet",
            keep_path=args.main_output_dir / f"{split}.parquet",
            overflow_path=args.overflow_output_dir / f"{split}.parquet",
            max_length=args.max_length,
        )

    (args.main_output_dir / "length_bucket_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (args.overflow_output_dir / "length_bucket_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
