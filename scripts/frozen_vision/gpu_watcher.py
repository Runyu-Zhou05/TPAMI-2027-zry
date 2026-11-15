#!/usr/bin/env python3
"""Wait for eight idle GPUs, then run one portable FVE queue."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fve_runtime import DEFAULT_OUTPUT_ROOT, DEFAULT_RAY_TMP_ROOT, ROOT, default_run_id, resolve_path, validate_run_id
from run_queue import validate_completed_marker


QUEUE = ROOT / "scripts/frozen_vision/run_queue.py"


def query_free(memory_limit, utilization_limit):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
    )
    try:
        process_output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        process_output = ""
    busy_uuids = {line.split(",", 1)[0].strip() for line in process_output.splitlines() if line.strip()}
    result = []
    for line in output.splitlines():
        index_text, uuid, memory_text, utilization_text = [value.strip() for value in line.split(",")]
        index, memory, utilization = int(index_text), int(memory_text), int(utilization_text)
        if uuid not in busy_uuids and memory < memory_limit and utilization <= utilization_limit:
            result.append(index)
    return result


def has_owned_ray_processes():
    result = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-f", "raylet|gcs_server|dashboard.py|monitor.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required", type=int, default=8, help="Use all eight H20 GPUs by default")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--memory-limit-mib", type=int, default=1024)
    parser.add_argument("--utilization-limit", type=int, default=5)
    parser.add_argument("--minimum-free-disk-gib", type=int, default=200)
    parser.add_argument("--output-root", default=os.environ.get("FVE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--run-id", default=os.environ.get("FVE_RUN_ID"))
    parser.add_argument("--model-path", default=os.environ.get("FVE_MODEL_PATH"), required=os.environ.get("FVE_MODEL_PATH") is None)
    parser.add_argument("--data-root", default=os.environ.get("FVE_DATA_ROOT", str(ROOT / "our_datasets")))
    parser.add_argument("--smoke-data-root", default=os.environ.get("FVE_SMOKE_DATA_ROOT"))
    parser.add_argument("--python", dest="python_executable", default=os.environ.get("FVE_PYTHON", sys.executable))
    parser.add_argument("--ray-tmp-root", default=os.environ.get("FVE_RAY_TMP_ROOT", str(DEFAULT_RAY_TMP_ROOT)))
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    run_id = validate_run_id(args.run_id or default_run_id())
    data_root = resolve_path(args.data_root)
    smoke_data_root = resolve_path(args.smoke_data_root or output_root / "smoke_data")
    ray_tmp_root = resolve_path(args.ray_tmp_root)
    state_dir = output_root / "state" / run_id
    complete_marker = state_dir / "queue.complete"
    failed_marker = state_dir / "queue.failed"
    lock_path = output_root / "state" / "watcher.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if complete_marker.exists():
        try:
            completed = json.loads(complete_marker.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(f"Queue completion marker is corrupt: {complete_marker}") from exc
        validate_completed_marker(completed, run_id)
        provenance = completed.get("invocation_provenance", {})
        if (
            Path(provenance.get("model_path", "")).expanduser().resolve() != resolve_path(args.model_path)
            or Path(provenance.get("output_root", "")).expanduser().resolve() != output_root
            or Path(provenance.get("ray_tmp_root", "")).expanduser().resolve() != ray_tmp_root
            or Path(provenance.get("data_manifest", {}).get("path", "")).parent != data_root / "MMBench_RotationQA_90degree"
        ):
            raise SystemExit(f"Queue completion marker belongs to a different watcher invocation: {complete_marker}")
        print(f"Queue already completed: {complete_marker}")
        return
    if failed_marker.exists():
        raise SystemExit(f"Previous queue failure requires manual review: {failed_marker}")

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("Another FVE watcher already holds the lock") from exc
    lock.write(str(os.getpid()))
    lock.flush()

    stable_group = None
    stable_count = 0
    while True:
        free_disk_gib = shutil.disk_usage(output_root).free / 2**30
        if free_disk_gib < args.minimum_free_disk_gib:
            raise SystemExit(f"Only {free_disk_gib:.1f} GiB free at {output_root}")
        free = query_free(args.memory_limit_mib, args.utilization_limit)
        ray_busy = has_owned_ray_processes()
        if ray_busy:
            free = []
        group = tuple(free[: args.required]) if len(free) >= args.required else None
        if group is not None and group == stable_group:
            stable_count += 1
        elif group is not None:
            stable_group = group
            stable_count = 1
        else:
            stable_group = None
            stable_count = 0
        print(
            f"free_gpus={free} candidate={stable_group} stable={stable_count}/{args.consecutive} "
            f"free_disk_gib={free_disk_gib:.1f} owned_ray_processes={ray_busy}",
            flush=True,
        )
        if stable_group is not None and stable_count >= args.consecutive:
            gpu_string = ",".join(map(str, stable_group))
            command = [
                args.python_executable,
                str(QUEUE),
                "--gpus", gpu_string,
                "--output-root", str(output_root),
                "--run-id", run_id,
                "--model-path", args.model_path,
                "--data-root", str(data_root),
                "--smoke-data-root", str(smoke_data_root),
                "--python", args.python_executable,
                "--ray-tmp-root", str(ray_tmp_root),
                "--minimum-free-disk-gib", str(args.minimum_free_disk_gib),
            ]
            print(f"Starting FVE queue on physical GPUs {gpu_string} for run {run_id}", flush=True)
            raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
