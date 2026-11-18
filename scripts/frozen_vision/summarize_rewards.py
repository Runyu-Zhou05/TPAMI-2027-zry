#!/usr/bin/env python3
"""Validate and summarize lossless raw reward traces for one stage."""

import argparse
import json
import os
import re
from collections import defaultdict
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
        from verl.utils.reward_difficulty import aggregate_group_statistics, group_reward_statistics
    except ImportError as exc:
        raise SystemExit("NumPy and PyArrow are required to summarize reward traces") from exc
    files = sorted(args.task_dir.glob("step_*.parquet"))
    if not files:
        raise SystemExit(f"No reward traces found in {args.task_dir}")

    rewards = []
    grouped_rewards = defaultdict(list)
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
                "uid",
                "rollout_id",
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
        for uid, rollout_id, reward in zip(
            table["uid"].to_pylist(),
            table["rollout_id"].to_pylist(),
            table["raw_reward"].to_pylist(),
            strict=True,
        ):
            grouped_rewards[(filename_step, str(uid))].append((int(rollout_id), float(reward)))
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
    group_metrics = []
    for identity, rollout_rewards in grouped_rewards.items():
        ordered = sorted(rollout_rewards)
        if [rollout_id for rollout_id, _ in ordered] != list(range(len(ordered))):
            raise SystemExit(f"Invalid rollout IDs for prompt group {identity}")
        group_metrics.append(group_reward_statistics([reward for _, reward in ordered]))
    difficulty = aggregate_group_statistics(group_metrics)
    summary = {
        "all_fail_group_fraction": difficulty["all_fail_group_fraction"],
        "all_success_group_fraction": difficulty["all_success_group_fraction"],
        "checkpoint_step": checkpoint_step,
        "expected_rows_per_step": args.expected_rows_per_step,
        "first_step": steps[0],
        "last_step": steps[-1],
        "max": float(values.max()),
        "informative_group_fraction": difficulty["informative_group_fraction"],
        "mean_group_reward_entropy_bits": difficulty["mean_group_reward_entropy_bits"],
        "mean_group_reward_variance": difficulty["mean_group_reward_variance"],
        "min": float(values.min()),
        "mixed_success_group_fraction": difficulty["mixed_success_group_fraction"],
        "prompt_groups": difficulty["prompt_groups"],
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
