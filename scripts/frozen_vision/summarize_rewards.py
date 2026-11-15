#!/usr/bin/env python3
"""Validate and summarize lossless raw reward traces for one stage."""

import argparse
import json
import os
import re
from pathlib import Path

REQUIRED_COLUMNS = {
    "run_id",
    "task",
    "global_step",
    "epoch",
    "reward_computed_at_utc_ns",
    "trace_written_at_utc_ns",
    "uid",
    "rollout_id",
    "dataset_index",
    "data_source",
    "ground_truth",
    "response",
    "raw_reward",
}


def committed_step(checkpoint_root: Path) -> int:
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise SystemExit(f"Missing committed-checkpoint tracker: {tracker}")
    try:
        step = int(tracker.read_text())
    except ValueError as exc:
        raise SystemExit(f"Malformed committed-checkpoint tracker: {tracker}") from exc
    manifest = checkpoint_root / f"global_step_{step}" / "checkpoint_complete.json"
    if not manifest.is_file():
        raise SystemExit(f"Missing checkpoint completion manifest: {manifest}")
    payload = json.loads(manifest.read_text())
    if int(payload.get("global_step", -1)) != step:
        raise SystemExit(f"Checkpoint manifest step disagrees with tracker: {manifest}")
    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument("--expected-rows-per-step", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("NumPy and PyArrow are required to summarize reward traces") from exc
    files = sorted(args.task_dir.glob("step_*.parquet"))
    if not files:
        raise SystemExit(f"No reward traces found in {args.task_dir}")

    rewards = []
    steps = []
    for file in files:
        match = re.fullmatch(r"step_(\d{8})\.parquet", file.name)
        if match is None:
            raise SystemExit(f"Unexpected reward trace filename: {file}")
        filename_step = int(match.group(1))
        parquet = pq.ParquetFile(file)
        columns = set(parquet.schema_arrow.names)
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise SystemExit(f"Reward trace {file} is missing columns: {missing}")
        if parquet.metadata.num_rows != args.expected_rows_per_step:
            raise SystemExit(
                f"Reward trace {file} has {parquet.metadata.num_rows} rows; "
                f"expected {args.expected_rows_per_step}"
            )
        table = pq.read_table(
            file,
            columns=[
                "run_id",
                "task",
                "global_step",
                "reward_computed_at_utc_ns",
                "trace_written_at_utc_ns",
                "raw_reward",
            ],
        )
        run_ids = set(table["run_id"].to_pylist())
        tasks = set(table["task"].to_pylist())
        embedded_steps = set(table["global_step"].to_pylist())
        if run_ids != {args.expected_run_id}:
            raise SystemExit(f"Reward trace {file} has unexpected run IDs: {run_ids}")
        if tasks != {args.expected_task}:
            raise SystemExit(f"Reward trace {file} has unexpected tasks: {tasks}")
        if embedded_steps != {filename_step}:
            raise SystemExit(f"Reward trace {file} embeds steps {embedded_steps}")
        computed = table["reward_computed_at_utc_ns"].to_numpy()
        written = table["trace_written_at_utc_ns"].to_numpy()
        if np.any(written < computed):
            raise SystemExit(f"Reward trace {file} has write timestamps before reward computation")
        rewards.append(table["raw_reward"].to_numpy())
        steps.append(filename_step)

    expected_steps = list(range(1, max(steps) + 1))
    if steps != expected_steps:
        raise SystemExit(f"Reward trace steps are not contiguous: {steps}")
    checkpoint_step = committed_step(args.checkpoint_root.expanduser().resolve())
    if steps[-1] != checkpoint_step:
        raise SystemExit(
            f"Final reward trace step {steps[-1]} does not match committed checkpoint {checkpoint_step}"
        )

    values = np.concatenate(rewards)
    summary = {
        "checkpoint_step": checkpoint_step,
        "expected_rows_per_step": args.expected_rows_per_step,
        "first_step": steps[0],
        "last_step": steps[-1],
        "max": float(values.max()),
        "min": float(values.min()),
        "raw_reward_mean": float(values.mean()),
        "raw_reward_std": float(values.std()),
        "reward_count": int(values.size),
        "run_id": args.expected_run_id,
        "steps": len(files),
        "task": args.expected_task,
        "trace_validated": True,
        "zero_rate": float(np.mean(values <= 0.0)),
    }
    output = args.task_dir / "summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
