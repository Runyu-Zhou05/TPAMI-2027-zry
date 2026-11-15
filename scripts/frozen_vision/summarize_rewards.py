#!/usr/bin/env python3
"""Summarize lossless raw reward traces for one task."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    args = parser.parse_args()
    files = sorted(args.task_dir.glob("step_*.parquet"))
    if not files:
        raise SystemExit(f"No reward traces found in {args.task_dir}")

    rewards = []
    steps = []
    for file in files:
        table = pq.read_table(file, columns=["global_step", "raw_reward"])
        rewards.append(table["raw_reward"].to_numpy())
        steps.append(int(table["global_step"][0].as_py()))
    values = np.concatenate(rewards)
    summary = {
        "task": args.task_dir.name,
        "steps": len(files),
        "first_step": min(steps),
        "last_step": max(steps),
        "reward_count": int(values.size),
        "raw_reward_mean": float(values.mean()),
        "raw_reward_std": float(values.std()),
        "zero_rate": float(np.mean(values <= 0.0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    output = args.task_dir / "summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
