#!/usr/bin/env python3
"""Run FVE smoke gates followed by the four training tasks."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fve_runtime import DEFAULT_OUTPUT_ROOT, DEFAULT_RAY_TMP_ROOT, ROOT, TASKS, default_run_id, resolve_path, stage_layout, validate_run_id


RUNNER = ROOT / "scripts/frozen_vision/run_frozen_vision.py"
SUMMARY = ROOT / "scripts/frozen_vision/summarize_rewards.py"


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

    if complete_marker.exists():
        print(f"Queue already completed: {complete_marker}")
        return
    if failed_marker.exists():
        raise SystemExit(f"Previous queue failure requires manual review: {failed_marker}")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpus
    stages = [
        ("position", "quick"),
        ("position", "quick-resume"),
        ("position", "production-smoke"),
    ] + [(task, "train") for task in TASKS]

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
            run([args.python_executable, str(SUMMARY), str(run_dir / "reward_traces" / trace_task)], environment, log_path)
    except BaseException as exc:
        failed_marker.write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    else:
        failed_marker.unlink(missing_ok=True)
        complete_marker.write_text("FVE_QUEUE_COMPLETE\n")
        print(f"FVE_QUEUE_COMPLETE: {complete_marker}")


if __name__ == "__main__":
    main()
