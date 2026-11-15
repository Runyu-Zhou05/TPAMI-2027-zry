#!/usr/bin/env python3
"""Resolve an FVE actor checkpoint and expose it as a validated HF model."""

from __future__ import annotations

import argparse
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


def global_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Not a global-step directory: {path}")
    return int(match.group(1))


def resolve_actor_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name == "actor":
        actor_dir = checkpoint
    elif checkpoint.name.startswith("global_step_") and (checkpoint / "actor").is_dir():
        actor_dir = checkpoint / "actor"
    else:
        candidates = [path for path in checkpoint.glob("global_step_*") if (path / "actor").is_dir()]
        if not candidates:
            raise SystemExit(f"No actor checkpoint found under {checkpoint}")
        actor_dir = max(candidates, key=global_step) / "actor"
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
    return {
        "architectures": getattr(config, "architectures", None),
        "model_type": config.model_type,
        "safetensor_files": [path.name for path in files],
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
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    actor_dir = resolve_actor_dir(args.checkpoint)
    step_dir = actor_dir.parent
    step = global_step(step_dir)
    direct_hf = actor_dir / "huggingface"
    direct_weights = list(direct_hf.glob("*.safetensors"))

    if direct_weights:
        model_dir = direct_hf
        export_mode = "direct_hf_checkpoint"
    else:
        model_dir = (args.target_dir or default_target(actor_dir)).expanduser().resolve()
        export_mode = "merged_fsdp_shards"
        if not model_dir.exists():
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
        "audit": audit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_mode": export_mode,
        "exported_model": str(model_dir),
        "generator_git_commit": git_commit(),
        "global_step": step,
    }
    manifest_path = model_dir / "fve_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_result(args.result_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
