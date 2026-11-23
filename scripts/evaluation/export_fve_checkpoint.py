#!/usr/bin/env python3
"""Resolve an SSL4RL actor checkpoint and expose it as a validated HF model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VISUAL_PARAMS = 668_684_288
EXPECTED_TOTAL_PARAMS = 3_754_622_976
MODEL_ROLES = ("position", "rotation", "jigsaw", "contrastive")
EXPORT_MANIFESTS = {
    "fve_export": "fve_export_manifest.json",
    "ssl4rl_export": "ssl4rl_export_manifest.json",
}


def global_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Not a global-step directory: {path}")
    return int(match.group(1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_step(step_dir: Path) -> dict:
    manifest_path = step_dir / "checkpoint_complete.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Checkpoint is not committed: {step_dir}")
    manifest = json.loads(manifest_path.read_text())
    step = global_step(step_dir)
    if manifest.get("schema_version") != 2 or int(manifest.get("global_step", -1)) != step:
        raise SystemExit(f"Checkpoint manifest step mismatch: {manifest_path}")
    for section, root_name in (("actor_files", "actor"), ("critic_files", "critic")):
        for entry in manifest.get(section, []):
            path = step_dir / root_name / entry["path"]
            if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
                raise SystemExit(f"Checkpoint manifest references an incomplete file: {path}")
            if sha256_file(path) != entry.get("sha256"):
                raise SystemExit(f"Checkpoint manifest hash mismatch: {path}")
    data_entry = manifest.get("dataloader")
    data_path = step_dir / data_entry["path"] if isinstance(data_entry, dict) else None
    if data_path is None or not data_path.is_file() or data_path.stat().st_size != int(data_entry["bytes"]):
        raise SystemExit(f"Checkpoint dataloader state is incomplete: {step_dir}")
    if sha256_file(data_path) != data_entry.get("sha256"):
        raise SystemExit(f"Checkpoint dataloader hash mismatch: {data_path}")
    return manifest


def resolve_actor_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name == "actor":
        step_dir = checkpoint.parent
    elif checkpoint.name.startswith("global_step_") and (checkpoint / "actor").is_dir():
        step_dir = checkpoint
    else:
        tracker = checkpoint / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise SystemExit(f"Committed-checkpoint tracker is missing: {tracker}")
        try:
            step = int(tracker.read_text())
        except ValueError as exc:
            raise SystemExit(f"Committed-checkpoint tracker is malformed: {tracker}") from exc
        step_dir = checkpoint / f"global_step_{step}"
    validate_checkpoint_step(step_dir)
    actor_dir = step_dir / "actor"
    if not (actor_dir / "huggingface" / "config.json").is_file():
        raise SystemExit(f"Checkpoint lacks Hugging Face config/processor files: {actor_dir / 'huggingface'}")
    return actor_dir


def weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = sorted(set(index.get("weight_map", {}).values()))
        files = [model_dir / name for name in names]
    else:
        files = sorted(model_dir.glob("*.safetensors"))
    missing = [path for path in files if not path.is_file()]
    if not files or missing:
        raise SystemExit(f"Incomplete Hugging Face weights under {model_dir}; missing={missing}")
    return files


def complete_hf_weights(model_dir: Path) -> bool:
    try:
        weight_files(model_dir)
        return True
    except SystemExit:
        return False


def audit_hf_model(model_dir: Path) -> dict:
    try:
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor
    except ImportError as exc:
        raise SystemExit("safetensors and transformers are required to validate the exported model") from exc

    config = AutoConfig.from_pretrained(model_dir)
    if getattr(config, "model_type", None) != "qwen2_5_vl":
        raise SystemExit(f"Unexpected exported model type: {getattr(config, 'model_type', None)!r}")
    AutoProcessor.from_pretrained(model_dir)

    total_params = 0
    visual_params = 0
    tensor_count = 0
    files = weight_files(model_dir)
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                numel = 1
                for dimension in shape:
                    numel *= dimension
                total_params += numel
                if key.startswith("visual."):
                    visual_params += numel
                tensor_count += 1
    if total_params != EXPECTED_TOTAL_PARAMS:
        raise SystemExit(f"Exported parameter count mismatch: {total_params} != {EXPECTED_TOTAL_PARAMS}")
    if visual_params != EXPECTED_VISUAL_PARAMS:
        raise SystemExit(f"Exported visual parameter count mismatch: {visual_params} != {EXPECTED_VISUAL_PARAMS}")
    support_files = {
        str(path.relative_to(model_dir)): sha256_file(path)
        for path in sorted(item for item in model_dir.rglob("*") if item.is_file())
        if path.suffix != ".safetensors" and path.name not in set(EXPORT_MANIFESTS.values())
    }
    return {
        "architectures": getattr(config, "architectures", None),
        "model_type": config.model_type,
        "safetensor_files": [path.name for path in files],
        "safetensor_sha256": {path.name: sha256_file(path) for path in files},
        "support_file_sha256": support_files,
        "tensor_count": tensor_count,
        "total_params": total_params,
        "visual_params": visual_params,
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def default_target(actor_dir: Path) -> Path:
    step_dir = actor_dir.parent
    checkpoint_root = step_dir.parent
    run_dir = checkpoint_root.parent
    return run_dir / "exports" / f"{step_dir.name}-hf"


def write_result(path: Path | None, payload: dict) -> None:
    if path is not None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--python", dest="python_executable", default=os.environ.get("FVE_PYTHON", sys.executable))
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--artifact-type", choices=tuple(EXPORT_MANIFESTS), default="fve_export")
    parser.add_argument("--model-role", choices=MODEL_ROLES, default="rotation")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.artifact_type == "fve_export" and args.model_role != "rotation":
        raise SystemExit("FVE exports are Rotation-only; use --artifact-type ssl4rl_export for paper-recipe tasks")
    export_manifest_name = EXPORT_MANIFESTS[args.artifact_type]

    actor_dir = resolve_actor_dir(args.checkpoint)
    step_dir = actor_dir.parent
    step = global_step(step_dir)
    completion_manifest_path = step_dir / "checkpoint_complete.json"
    completion_manifest_sha256 = sha256_file(completion_manifest_path)
    direct_hf = actor_dir / "huggingface"

    if complete_hf_weights(direct_hf):
        model_dir = direct_hf
        export_mode = "direct_hf_checkpoint"
    else:
        shard_files = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
        if not shard_files:
            raise SystemExit(f"Checkpoint has neither complete HF weights nor FSDP model shards: {actor_dir}")
        model_dir = (args.target_dir or default_target(actor_dir)).expanduser().resolve()
        export_mode = "merged_fsdp_shards"
        if model_dir.exists():
            existing_manifest_path = model_dir / export_manifest_name
            if not existing_manifest_path.is_file():
                raise SystemExit(f"Refusing to reuse unbound export directory: {model_dir}")
            existing_manifest = json.loads(existing_manifest_path.read_text())
            if (
                existing_manifest.get("actor_checkpoint") != str(actor_dir)
                or existing_manifest.get("source_checkpoint_manifest_sha256") != completion_manifest_sha256
                or existing_manifest.get("artifact_type", "fve_export") != args.artifact_type
                or existing_manifest.get("model_role", "rotation") != args.model_role
            ):
                raise SystemExit(f"Existing export is bound to a different checkpoint: {model_dir}")
        else:
            if args.verify_only:
                raise SystemExit(f"No exported model found at {model_dir}")
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{model_dir.name}.building-", dir=model_dir.parent) as temp:
                staging = Path(temp)
                command = [
                    args.python_executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    str(actor_dir),
                    "--target_dir",
                    str(staging),
                    "--use_cpu_initialization",
                ]
                subprocess.run(command, cwd=ROOT, check=True)
                audit_hf_model(staging)
                os.replace(staging, model_dir)

    audit = audit_hf_model(model_dir)
    manifest = {
        "actor_checkpoint": str(actor_dir),
        "artifact_type": args.artifact_type,
        "audit": audit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_mode": export_mode,
        "exported_model": str(model_dir),
        "generator_git_commit": git_commit(),
        "global_step": step,
        "model_role": args.model_role,
        "schema_version": 1,
        "source_checkpoint_manifest_sha256": completion_manifest_sha256,
    }
    manifest_path = model_dir / export_manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_result(args.result_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
