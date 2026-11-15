"""Shared portable runtime configuration for Frozen Vision Encoder (FVE)."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv")
DEFAULT_RAY_TMP_ROOT = Path("/tmp/fve-ray")
EXPECTED_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_hf_download_file(model_path: Path, relative_path: str) -> dict:
    path = model_path / relative_path
    metadata = model_path / ".cache" / "huggingface" / "download" / f"{relative_path}.metadata"
    if not path.is_file() or not metadata.is_file():
        raise SystemExit(f"Base model snapshot file or metadata is missing: {path}")
    lines = metadata.read_text().splitlines()
    if len(lines) < 2 or lines[0].strip() != EXPECTED_MODEL_REVISION:
        raise SystemExit(f"Base model snapshot metadata has the wrong revision: {metadata}")
    etag = lines[1].strip().removeprefix("W/").strip('"')
    sha256 = _sha256_file(path)
    git_blob_sha1 = None
    if etag != sha256:
        git_blob_sha1 = _git_blob_sha1(path)
        if etag != git_blob_sha1:
            raise SystemExit(f"Base model file no longer matches its Hugging Face metadata ETag: {path}")
    return {
        "bytes": path.stat().st_size,
        "etag": etag,
        "git_blob_sha1": git_blob_sha1,
        "path": str(path),
        "sha256": sha256,
    }


def validate_base_model_identity(model_path: Path) -> dict:
    model_path = resolve_path(model_path)
    config_path = model_path / "config.json"
    config_record = _validate_hf_download_file(model_path, "config.json")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "qwen2_5_vl" or "Qwen2_5_VLForConditionalGeneration" not in config.get(
        "architectures", []
    ):
        raise SystemExit(f"Unexpected base-model architecture in {config_path}")
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"Pinned base model lacks its safetensors index: {index_path}")
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if len(shard_names) != 2 or any(not re.fullmatch(r"model-\d{5}-of-00002\.safetensors", name) for name in shard_names):
        raise SystemExit(f"Pinned base model has an unexpected safetensors shard set: {shard_names}")
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    tracked_names = sorted(
        str(path.relative_to(metadata_root))[: -len(".metadata")]
        for path in metadata_root.rglob("*.metadata")
    )
    required_names = {"config.json", "model.safetensors.index.json", *shard_names}
    if not required_names.issubset(tracked_names):
        raise SystemExit(f"Base model snapshot metadata is incomplete: {sorted(required_names - set(tracked_names))}")
    files = {name: _validate_hf_download_file(model_path, name) for name in tracked_names}
    return {
        "config": config_path,
        "files": files,
        "metadata": model_path / ".cache" / "huggingface" / "download" / "config.json.metadata",
        "revision": EXPECTED_MODEL_REVISION,
    }


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
