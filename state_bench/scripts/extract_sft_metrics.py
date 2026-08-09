"""Extract verl SFT console metrics from a log file into CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


STEP_RE = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")


def parse_value(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def parse_line(line: str) -> dict[str, float | str] | None:
    match = STEP_RE.search(line)
    if not match:
        return None
    row: dict[str, float | str] = {"step": int(match.group("step"))}
    for part in match.group("body").split(" - "):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        row[key.strip()] = parse_value(value.strip())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for line in args.log_file.read_text(errors="replace").splitlines() if (row := parse_line(line))]
    fieldnames = ["step"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
