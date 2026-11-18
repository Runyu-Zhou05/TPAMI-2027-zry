#!/usr/bin/env python3
"""Turn lossless rollout traces into reviewer-facing GRPO difficulty metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRIC_SPEC = importlib.util.spec_from_file_location(
    "fve_reward_difficulty_metrics", ROOT / "verl/utils/reward_difficulty.py"
)
METRIC_MODULE = importlib.util.module_from_spec(METRIC_SPEC)
assert METRIC_SPEC.loader is not None
METRIC_SPEC.loader.exec_module(METRIC_MODULE)
aggregate_group_statistics = METRIC_MODULE.aggregate_group_statistics
group_reward_statistics = METRIC_MODULE.group_reward_statistics


REQUIRED_COLUMNS = {
    "run_id",
    "task",
    "global_step",
    "epoch",
    "uid",
    "rollout_id",
    "dataset_index",
    "data_source",
    "ground_truth",
    "raw_reward",
}
STAGE_NAMES = ("early", "middle", "late")


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    columns = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def assign_stages(steps: list[int]) -> dict[int, str]:
    import numpy as np

    chunks = np.array_split(np.asarray(steps, dtype=np.int64), len(STAGE_NAMES))
    return {int(step): name for name, chunk in zip(STAGE_NAMES, chunks) for step in chunk.tolist()}


def trend_summary(step_records: list[dict], field: str) -> dict:
    import numpy as np

    values = np.asarray([record[field] for record in step_records], dtype=np.float64)
    if len(values) == 1:
        return {"auc_over_normalized_progress": float(values[0]), "linear_slope_per_progress": 0.0}
    progress = np.linspace(0.0, 1.0, num=len(values))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return {
        "auc_over_normalized_progress": float(trapezoid(values, progress)),
        "linear_slope_per_progress": float(np.polyfit(progress, values, deg=1)[0]),
    }


def read_groups(
    trace_files: list[Path],
    *,
    expected_run_id: str | None,
    expected_task: str | None,
    expected_rollouts: int | None,
    variance_epsilon: float,
    success_threshold: float,
    reward_round_decimals: int,
) -> tuple[list[dict], dict]:
    import numpy as np
    import pyarrow.parquet as pq

    group_records = []
    run_ids = set()
    tasks = set()
    input_records = {}
    observed_steps = []
    for trace_file in trace_files:
        match = re.fullmatch(r"step_(\d{8})\.parquet", trace_file.name)
        if match is None:
            raise SystemExit(f"Unexpected trace filename: {trace_file}")
        filename_step = int(match.group(1))
        parquet = pq.ParquetFile(trace_file)
        missing = REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
        if missing:
            raise SystemExit(f"Trace lacks required columns {sorted(missing)}: {trace_file}")
        table = pq.read_table(trace_file, columns=sorted(REQUIRED_COLUMNS))
        payload = table.to_pydict()
        file_steps = {int(value) for value in payload["global_step"]}
        epochs = {int(value) for value in payload["epoch"]}
        if file_steps != {filename_step} or len(epochs) != 1:
            raise SystemExit(f"Trace step/epoch identity mismatch: {trace_file}")
        file_run_ids = {str(value) for value in payload["run_id"]}
        file_tasks = {str(value) for value in payload["task"]}
        if len(file_run_ids) != 1 or len(file_tasks) != 1:
            raise SystemExit(f"Trace mixes run or task identities: {trace_file}")
        run_ids.update(file_run_ids)
        tasks.update(file_tasks)
        epoch = next(iter(epochs))
        by_uid = defaultdict(list)
        for row in range(table.num_rows):
            uid = payload["uid"][row]
            if uid in (None, "", "None"):
                raise SystemExit(f"Trace contains an empty prompt UID: {trace_file}")
            reward = float(payload["raw_reward"][row])
            if not np.isfinite(reward):
                raise SystemExit(f"Trace contains a non-finite reward: {trace_file}")
            by_uid[str(uid)].append(
                {
                    "data_source": payload["data_source"][row],
                    "dataset_index": payload["dataset_index"][row],
                    "ground_truth": payload["ground_truth"][row],
                    "reward": reward,
                    "rollout_id": int(payload["rollout_id"][row]),
                }
            )
        for uid, rows in by_uid.items():
            rollout_ids = sorted(row["rollout_id"] for row in rows)
            if rollout_ids != list(range(len(rows))):
                raise SystemExit(f"Prompt {uid} has invalid rollout IDs at step {filename_step}: {rollout_ids}")
            if expected_rollouts is not None and len(rows) != expected_rollouts:
                raise SystemExit(
                    f"Prompt {uid} has {len(rows)} rollouts at step {filename_step}; expected {expected_rollouts}"
                )
            for identity_field in ("data_source", "dataset_index", "ground_truth"):
                values = {str(row[identity_field]) for row in rows}
                if len(values) != 1:
                    raise SystemExit(f"Prompt {uid} mixes {identity_field} values: {values}")
            metrics = group_reward_statistics(
                [row["reward"] for row in rows],
                variance_epsilon=variance_epsilon,
                success_threshold=success_threshold,
                reward_round_decimals=reward_round_decimals,
            )
            group_records.append(
                {
                    "run_id": next(iter(file_run_ids)),
                    "task": next(iter(file_tasks)),
                    "global_step": filename_step,
                    "epoch": epoch,
                    "uid": uid,
                    "dataset_index": str(rows[0]["dataset_index"]),
                    "data_source": str(rows[0]["data_source"]),
                    "ground_truth": str(rows[0]["ground_truth"]),
                    **metrics,
                }
            )
        group_sizes = {len(rows) for rows in by_uid.values()}
        if len(group_sizes) != 1:
            raise SystemExit(f"Trace has inconsistent rollout group sizes {group_sizes}: {trace_file}")
        observed_steps.append(filename_step)
        input_records[trace_file.name] = {
            "bytes": trace_file.stat().st_size,
            "rows": table.num_rows,
            "sha256": sha256_file(trace_file),
        }

    if len(run_ids) != 1 or len(tasks) != 1:
        raise SystemExit(f"Trace directory mixes runs/tasks: run_ids={run_ids}, tasks={tasks}")
    run_id = next(iter(run_ids))
    task = next(iter(tasks))
    if expected_run_id is not None and run_id != expected_run_id:
        raise SystemExit(f"Trace run ID {run_id!r} != expected {expected_run_id!r}")
    if expected_task is not None and task != expected_task:
        raise SystemExit(f"Trace task {task!r} != expected {expected_task!r}")
    expected_steps = list(range(min(observed_steps), max(observed_steps) + 1))
    if observed_steps != expected_steps:
        raise SystemExit(f"Trace steps are not contiguous: {observed_steps}")
    return group_records, {
        "first_step": observed_steps[0],
        "input_files": input_records,
        "last_step": observed_steps[-1],
        "run_id": run_id,
        "task": task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-run-id", default=None)
    parser.add_argument("--expected-task", default=None)
    parser.add_argument("--expected-rollouts-per-prompt", type=int, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--variance-epsilon", type=float, default=1e-12)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--reward-round-decimals", type=int, default=8)
    args = parser.parse_args()
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:
        raise SystemExit("NumPy and PyArrow are required for reward difficulty analysis") from exc

    task_dir = args.task_dir.expanduser().resolve()
    trace_files = sorted(task_dir.glob("step_*.parquet"))
    if not trace_files:
        raise SystemExit(f"No reward trace Parquets found: {task_dir}")
    groups, identity = read_groups(
        trace_files,
        expected_run_id=args.expected_run_id,
        expected_task=args.expected_task,
        expected_rollouts=args.expected_rollouts_per_prompt,
        variance_epsilon=args.variance_epsilon,
        success_threshold=args.success_threshold,
        reward_round_decimals=args.reward_round_decimals,
    )
    if args.checkpoint_root is not None:
        from summarize_rewards import committed_step

        checkpoint_step = committed_step(args.checkpoint_root.expanduser().resolve())
        if checkpoint_step != identity["last_step"]:
            raise SystemExit(
                f"Final trace step {identity['last_step']} does not match committed checkpoint {checkpoint_step}"
            )
    else:
        checkpoint_step = None

    steps = sorted({int(record["global_step"]) for record in groups})
    stage_map = assign_stages(steps)
    for record in groups:
        record["stage"] = stage_map[int(record["global_step"])]
    by_step = defaultdict(list)
    by_stage = defaultdict(list)
    for record in groups:
        by_step[int(record["global_step"])].append(record)
        by_stage[record["stage"]].append(record)
    step_records = []
    for rank, step in enumerate(steps):
        records = by_step[step]
        step_records.append(
            {
                "global_step": step,
                "epoch": int(records[0]["epoch"]),
                "stage": stage_map[step],
                "normalized_progress": rank / (len(steps) - 1) if len(steps) > 1 else 1.0,
                **aggregate_group_statistics(records),
            }
        )
    stage_records = []
    for stage in (*STAGE_NAMES, "overall"):
        records = groups if stage == "overall" else by_stage.get(stage, [])
        if records:
            stage_records.append({"stage": stage, **aggregate_group_statistics(records)})

    config = {
        "expected_rollouts_per_prompt": args.expected_rollouts_per_prompt,
        "reward_entropy": "empirical categorical entropy in bits after deterministic reward rounding",
        "reward_round_decimals": args.reward_round_decimals,
        "stage_partition": "contiguous observed steps split into three equal-count early/middle/late chunks",
        "success_threshold": args.success_threshold,
        "variance_epsilon": args.variance_epsilon,
    }
    fingerprint_payload = {
        "config": config,
        "input_files": identity["input_files"],
        "metric_module_sha256": sha256_file(ROOT / "verl/utils/reward_difficulty.py"),
        "run_id": identity["run_id"],
        "script_sha256": sha256_file(Path(__file__)),
        "task": identity["task"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_name = "difficulty_analysis_manifest.json"
    if output_dir.exists():
        manifest_path = output_dir / manifest_name
        if not manifest_path.is_file():
            raise SystemExit(f"Refusing to reuse unbound analysis directory: {output_dir}")
        existing = json.loads(manifest_path.read_text())
        if existing.get("analysis_fingerprint") != fingerprint:
            raise SystemExit(f"Reward difficulty analysis fingerprint mismatch: {output_dir}")
        for name, expected_hash in existing.get("output_sha256", {}).items():
            path = output_dir / name
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise SystemExit(f"Reward difficulty output changed: {path}")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        groups_path = staging / "reward_difficulty_groups.parquet"
        pq.write_table(pa.Table.from_pylist(groups), groups_path, compression="zstd")
        write_csv(staging / "reward_difficulty_by_step.csv", step_records)
        write_csv(staging / "reward_difficulty_by_stage.csv", stage_records)
        overall = next(record for record in stage_records if record["stage"] == "overall")
        summary = {
            "analysis_fingerprint": fingerprint,
            "checkpoint_step": checkpoint_step,
            "config": config,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "first_step": identity["first_step"],
            "input_files": identity["input_files"],
            "last_step": identity["last_step"],
            "overall": overall,
            "run_id": identity["run_id"],
            "schema_version": 1,
            "task": identity["task"],
            "trends": {
                field: trend_summary(step_records, field)
                for field in (
                    "informative_group_fraction",
                    "mean_group_reward_entropy_bits",
                    "success_rate",
                )
            },
        }
        (staging / "reward_difficulty_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        output_hashes = {
            path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()
        }
        manifest = {
            **summary,
            "output_sha256": output_hashes,
            "source_metric_module_sha256": fingerprint_payload["metric_module_sha256"],
            "source_script_sha256": fingerprint_payload["script_sha256"],
        }
        (staging / manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
