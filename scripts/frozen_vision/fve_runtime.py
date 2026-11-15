"""Shared portable runtime configuration for Frozen Vision Encoder (FVE)."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv")
DEFAULT_RAY_TMP_ROOT = Path("/tmp/fve-ray")

TASKS = {
    "rotation": ("MMBench_RotationQA_90degree", "ssl4rl_qwen_3b_mmbench_rotation_frozen_vision"),
}


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def default_run_id(prefix: str = "fve") -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("run ID must contain only letters, digits, '.', '_' or '-' and be at most 80 characters")
    return value


def mode_suffix(mode: str) -> str:
    if mode in {"quick", "quick-resume"}:
        return "_quick_smoke"
    if mode == "production-smoke":
        return "_production_smoke"
    if mode == "train":
        return ""
    raise ValueError(f"Unknown FVE mode: {mode}")


def stage_layout(output_root: Path, run_id: str, task: str, mode: str) -> tuple[str, Path, str]:
    _, base_experiment = TASKS[task]
    suffix = mode_suffix(mode)
    experiment = base_experiment + suffix
    trace_task = task + suffix
    return experiment, output_root / "runs" / run_id / experiment, trace_task


def short_ray_tmpdir(ray_tmp_root: Path, run_id: str, experiment: str) -> Path:
    """Keep Ray's UNIX-domain socket path safely below its 107-byte limit."""
    token = hashlib.sha256(f"{run_id}:{experiment}".encode()).hexdigest()[:16]
    return ray_tmp_root / f"fve-{token}"
