#!/usr/bin/env python3
"""Run Rotation FVE smoke gates followed by Rotation training."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from check_environment import validate_environment
from fve_runtime import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAY_TMP_ROOT,
    ROOT,
    default_run_id,
    resolve_path,
    stage_layout,
    validate_base_model_identity,
    validate_run_id,
)


RUNNER = ROOT / "scripts/frozen_vision/run_frozen_vision.py"
SUMMARY = ROOT / "scripts/frozen_vision/summarize_rewards.py"
ENVIRONMENT_CHECK = ROOT / "scripts/frozen_vision/check_environment.py"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def required_file(path, label):
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": sha256_file(path)}


def validate_file_record(record, label):
    if not isinstance(record, dict):
        raise SystemExit(f"Queue completion marker lacks {label}")
    path = Path(record.get("path", "")).expanduser().resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise SystemExit(f"Queue completion marker references changed {label}: {path}")
    return path


def validate_completed_marker(completed, run_id, expected_fingerprint=None):
    if (
        completed.get("schema_version") != 1
        or completed.get("status") != "complete"
        or completed.get("run_id") != run_id
        or (expected_fingerprint is not None and completed.get("invocation_fingerprint") != expected_fingerprint)
    ):
        raise SystemExit("Queue completion marker identity does not match this invocation")
    stages = completed.get("stages", [])
    if [stage.get("mode") for stage in stages] != ["quick", "quick-resume", "production-smoke", "train"]:
        raise SystemExit("Queue completion marker has an incomplete stage sequence")
    for stage in stages:
        if stage.get("task") != "rotation":
            raise SystemExit("Queue completion marker contains a non-Rotation stage")
        validate_file_record(stage.get("checkpoint", {}).get("completion_manifest"), "checkpoint manifest")
        validate_file_record(stage.get("checkpoint", {}).get("tracker"), "checkpoint tracker")
        validate_file_record(stage.get("reward_summary"), "reward summary")
    return completed


def invocation_provenance(args, output_root, run_id, data_root, smoke_data_root, ray_tmp_root, stages):
    model_path = resolve_path(args.model_path)
    model_identity = validate_base_model_identity(model_path)
    dataset_dir = data_root / "MMBench_RotationQA_90degree"
    provenance = {
        "data_manifest": required_file(dataset_dir / "manifest.json", "Rotation data manifest"),
        "environment": validate_environment(),
        "gpus": [value.strip() for value in args.gpus.split(",") if value.strip()],
        "minimum_free_disk_gib": args.minimum_free_disk_gib,
        "model_config": required_file(model_path / "config.json", "base-model config"),
        "model_revision_metadata": required_file(model_identity["metadata"], "base-model revision metadata"),
        "model_snapshot": model_identity["files"],
        "model_path": str(model_path),
        "output_root": str(output_root),
        "python": str(resolve_path(args.python_executable)),
        "ray_tmp_root": str(ray_tmp_root),
        "run_id": run_id,
        "scripts": {
            path.name: required_file(path, f"workflow script {path.name}")
            for path in (ENVIRONMENT_CHECK, RUNNER, SUMMARY, Path(__file__).resolve())
        },
        "smoke_files": {
            name: required_file(smoke_data_root / name, f"smoke dataset {name}")
            for name in ("quick.parquet", "production_shape.parquet")
        },
        "stages": [{"mode": mode, "task": task} for task, mode in stages],
    }
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return fingerprint, provenance


def checkpoint_record(checkpoint_root, mode):
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    required_file(tracker, "checkpoint tracker")
    try:
        step = int(tracker.read_text())
    except ValueError as exc:
        raise SystemExit(f"Malformed checkpoint tracker: {tracker}") from exc
    stage_tracker = tracker.with_name(f"checkpoint_stage.{mode}.json")
    atomic_json(stage_tracker, {"global_step": step, "mode": mode})
    tracker_record = required_file(stage_tracker, "stage checkpoint tracker")
    manifest_path = checkpoint_root / f"global_step_{step}" / "checkpoint_complete.json"
    return {
        "completion_manifest": required_file(manifest_path, "checkpoint completion manifest"),
        "global_step": step,
        "root": str(checkpoint_root),
        "tracker": tracker_record,
    }


def run(command, environment, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"COMMAND: {' '.join(map(str, command))}\n")
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise SystemExit(f"Command failed with status {process.returncode}; see {log_path}")


def cleanup_ray(python_executable, environment, log_path):
    python_path = Path(python_executable).expanduser()
    if not python_path.is_absolute():
        resolved_python = shutil.which(python_executable, path=environment.get("PATH"))
        if resolved_python is None:
            raise RuntimeError(f"Cannot resolve Python executable: {python_executable}")
        python_path = Path(resolved_python)
    ray_executable = python_path.parent / "ray"
    if not ray_executable.is_file() or not os.access(ray_executable, os.X_OK):
        raise RuntimeError(f"Ray CLI is not installed next to the selected Python: {ray_executable}")
    run([str(ray_executable), "stop", "--force"], environment, log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--output-root", default=os.environ.get("FVE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--run-id", default=os.environ.get("FVE_RUN_ID"))
    parser.add_argument("--model-path", default=os.environ.get("FVE_MODEL_PATH"), required=os.environ.get("FVE_MODEL_PATH") is None)
    parser.add_argument("--data-root", default=os.environ.get("FVE_DATA_ROOT", str(ROOT / "our_datasets")))
    parser.add_argument("--smoke-data-root", default=os.environ.get("FVE_SMOKE_DATA_ROOT"))
    parser.add_argument("--python", dest="python_executable", default=os.environ.get("FVE_PYTHON", sys.executable))
    parser.add_argument("--ray-tmp-root", default=os.environ.get("FVE_RAY_TMP_ROOT", str(DEFAULT_RAY_TMP_ROOT)))
    parser.add_argument("--minimum-free-disk-gib", type=int, default=200)
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    run_id = validate_run_id(args.run_id or default_run_id())
    data_root = resolve_path(args.data_root)
    smoke_data_root = resolve_path(args.smoke_data_root or output_root / "smoke_data")
    ray_tmp_root = resolve_path(args.ray_tmp_root)
    state_dir = output_root / "state" / run_id
    complete_marker = state_dir / "queue.complete"
    failed_marker = state_dir / "queue.failed"
    logs = output_root / "runs" / run_id / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpus
    stages = [
        ("rotation", "quick"),
        ("rotation", "quick-resume"),
        ("rotation", "production-smoke"),
        ("rotation", "train"),
    ]
    fingerprint, provenance = invocation_provenance(
        args, output_root, run_id, data_root, smoke_data_root, ray_tmp_root, stages
    )

    if complete_marker.exists():
        try:
            completed = json.loads(complete_marker.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(f"Queue completion marker is corrupt: {complete_marker}") from exc
        validate_completed_marker(completed, run_id, fingerprint)
        print(f"Queue already completed and validated: {complete_marker}")
        return
    if failed_marker.exists():
        raise SystemExit(f"Previous queue failure requires manual review: {failed_marker}")

    started_at = datetime.now(timezone.utc).isoformat()
    stage_records = []

    try:
        for task, mode in stages:
            free_disk_gib = shutil.disk_usage(output_root).free / 2**30
            if free_disk_gib < args.minimum_free_disk_gib:
                raise RuntimeError(f"Refusing to start {task}/{mode}: only {free_disk_gib:.1f} GiB free at {output_root}")
            log_path = logs / f"{task}_{mode}.log"
            runner_command = [
                args.python_executable,
                str(RUNNER),
                "--task", task,
                "--mode", mode,
                "--gpus", args.gpus,
                "--output-root", str(output_root),
                "--run-id", run_id,
                "--model-path", args.model_path,
                "--data-root", str(data_root),
                "--smoke-data-root", str(smoke_data_root),
                "--python", args.python_executable,
                "--ray-tmp-root", str(ray_tmp_root),
            ]
            try:
                run(runner_command, environment, log_path)
            finally:
                cleanup_ray(args.python_executable, environment, logs / "ray_cleanup.log")
            _, run_dir, trace_task = stage_layout(output_root, run_id, task, mode)
            expected_rows = 2 * len([value for value in args.gpus.split(",") if value.strip()])
            if mode not in {"quick", "quick-resume"}:
                expected_rows = 2560
            run(
                [
                    args.python_executable,
                    str(SUMMARY),
                    str(run_dir / "reward_traces" / trace_task),
                    "--expected-run-id",
                    run_id,
                    "--expected-task",
                    trace_task,
                    "--expected-rows-per-step",
                    str(expected_rows),
                    "--checkpoint-root",
                    str(run_dir / "checkpoints"),
                ],
                environment,
                log_path,
            )
            summary_path = run_dir / "reward_traces" / trace_task / "summary.json"
            stage_summary_path = summary_path.with_name(f"summary.{mode}.json")
            try:
                summary_payload = json.loads(summary_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                raise SystemExit(f"Malformed reward summary: {summary_path}") from exc
            atomic_json(stage_summary_path, summary_payload)
            stage_records.append(
                {
                    "checkpoint": checkpoint_record(run_dir / "checkpoints", mode),
                    "mode": mode,
                    "reward_summary": required_file(stage_summary_path, "stage reward summary"),
                    "run_dir": str(run_dir),
                    "task": task,
                }
            )
    except BaseException as exc:
        atomic_json(
            failed_marker,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "invocation_fingerprint": fingerprint,
                "run_id": run_id,
                "schema_version": 1,
                "stages_completed": stage_records,
                "status": "failed",
            },
        )
        raise
    else:
        failed_marker.unlink(missing_ok=True)
        atomic_json(
            complete_marker,
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "invocation_fingerprint": fingerprint,
                "invocation_provenance": provenance,
                "run_id": run_id,
                "schema_version": 1,
                "stages": stage_records,
                "started_at_utc": started_at,
                "status": "complete",
            },
        )
        print(f"FVE_QUEUE_COMPLETE: {complete_marker}")


if __name__ == "__main__":
    main()
