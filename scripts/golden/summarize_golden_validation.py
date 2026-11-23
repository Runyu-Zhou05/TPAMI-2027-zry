#!/usr/bin/env python3
"""Validate and summarize Golden-RLVR validation generation dumps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path


DEFAULT_EXPECTED_STEPS = (0, 20, 40, 60, 80, 100)
FIELDNAMES = ("step", "accuracy", "correct", "rows")


def parse_expected_steps(values: list[str]) -> tuple[int, ...]:
    """Accept comma-separated steps, whitespace-separated steps, or both."""
    parts = [part.strip() for value in values for part in value.split(",") if part.strip()]
    try:
        steps = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--expected-steps must contain integers") from exc
    if not steps:
        raise argparse.ArgumentTypeError("--expected-steps must not be empty")
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("--expected-steps must be non-negative")
    if any(current >= following for current, following in zip(steps, steps[1:])):
        raise argparse.ArgumentTypeError("--expected-steps must be unique and strictly increasing")
    return steps


def load_step(path: Path, expected_step: int) -> tuple[int, int]:
    """Return ``(correct, rows)`` after validating one generation dump."""
    if not path.is_file():
        raise SystemExit(f"Missing Golden validation dump for step {expected_step}: {path}")

    correct = 0
    rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SystemExit(f"Blank JSONL row at {path}:{line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise SystemExit(f"JSONL row is not an object at {path}:{line_number}")

                step = record.get("step")
                if isinstance(step, bool) or not isinstance(step, int) or step != expected_step:
                    raise SystemExit(
                        f"Step field mismatch at {path}:{line_number}: expected {expected_step}, got {step!r}"
                    )

                score = record.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise SystemExit(f"Score must be numeric 0 or 1 at {path}:{line_number}: {score!r}")
                numeric_score = float(score)
                if not math.isfinite(numeric_score) or numeric_score not in (0.0, 1.0):
                    raise SystemExit(f"Score must be finite 0 or 1 at {path}:{line_number}: {score!r}")

                correct += int(numeric_score)
                rows += 1
    except UnicodeError as exc:
        raise SystemExit(f"Golden validation dump is not valid UTF-8: {path}") from exc

    if rows == 0:
        raise SystemExit(f"Golden validation dump is empty: {path}")
    return correct, rows


def summarize(validation_data_dir: Path, expected_steps: tuple[int, ...]) -> list[dict[str, int | float]]:
    summaries = []
    expected_rows: int | None = None
    for step in expected_steps:
        correct, rows = load_step(validation_data_dir / f"{step}.jsonl", step)
        if expected_rows is None:
            expected_rows = rows
        elif rows != expected_rows:
            raise SystemExit(
                f"Golden validation row-count mismatch at step {step}: expected {expected_rows}, got {rows}"
            )
        summaries.append(
            {
                "step": step,
                "accuracy": correct / rows,
                "correct": correct,
                "rows": rows,
            }
        )
    return summaries


def publish(output_dir: Path, summaries: list[dict[str, int | float]]) -> None:
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse Golden validation summary directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.building-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        with (staging / "golden_validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(summaries)
        (staging / "golden_validation_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        required=True,
        help="trainer.validation_data_dir containing <step>.jsonl files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-steps",
        nargs="+",
        default=[",".join(str(step) for step in DEFAULT_EXPECTED_STEPS)],
        metavar="STEP",
        help="Expected steps, supplied comma-separated and/or space-separated",
    )
    args = parser.parse_args()

    try:
        expected_steps = parse_expected_steps(args.expected_steps)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    validation_data_dir = args.validation_data_dir.expanduser().resolve()
    if not validation_data_dir.is_dir():
        raise SystemExit(f"Golden validation data directory is missing: {validation_data_dir}")
    output_dir = args.output_dir.expanduser().resolve()
    summaries = summarize(validation_data_dir, expected_steps)
    publish(output_dir, summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
