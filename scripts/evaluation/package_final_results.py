#!/usr/bin/env python3
"""Package paper tables, evaluation evidence, and training manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--short"], capture_output=True, text=True, check=False
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()),
        "status": status.stdout.splitlines(),
    }


def inventory_tree(root: Path, hash_limit_bytes: int = 512 * 1024 * 1024) -> list[dict]:
    if not root.exists():
        return []
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        files.append(
            {
                "bytes": size,
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "sha256": sha256_file(path) if size <= hash_limit_bytes else None,
            }
        )
    return files


def copy_curated_training(training_dir: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    patterns = ("*.log", "summary.json", "*manifest.json", "latest_checkpointed_iteration.txt")
    copied = set()
    for pattern in patterns:
        for source in training_dir.rglob(pattern):
            if not source.is_file() or source in copied:
                continue
            relative = source.relative_to(training_dir)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.add(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--base-eval-manifest", type=Path, required=True)
    parser.add_argument("--rotation-eval-manifest", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not args.run_id or any(character not in allowed for character in args.run_id):
        raise SystemExit("--run-id contains unsupported characters")
    training_dir = args.training_run_dir.expanduser().resolve()
    base_manifest = args.base_eval_manifest.expanduser().resolve()
    rotation_manifest = args.rotation_eval_manifest.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (training_dir, base_manifest, rotation_manifest, tables_dir):
        if not path.exists():
            raise SystemExit(f"Required result path is missing: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"fve_{args.run_id}_results.tar.gz"
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if archive.exists():
        raise SystemExit(f"Refusing to overwrite result archive: {archive}")

    base_eval_dir = base_manifest.parent
    rotation_eval_dir = rotation_manifest.parent
    manifest = {
        "base_evaluation": inventory_tree(base_eval_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "rotation_evaluation": inventory_tree(rotation_eval_dir),
        "run_id": args.run_id,
        "tables": inventory_tree(tables_dir),
        "training": inventory_tree(training_dir),
    }
    with tempfile.TemporaryDirectory(prefix=f".fve-{args.run_id}-bundle-", dir=output_dir) as temporary:
        bundle = Path(temporary) / f"fve_{args.run_id}_results"
        bundle.mkdir()
        copy_curated_training(training_dir, bundle / "training")
        shutil.copytree(base_eval_dir, bundle / "evaluation" / "base", dirs_exist_ok=True)
        shutil.copytree(rotation_eval_dir, bundle / "evaluation" / "rotation", dirs_exist_ok=True)
        shutil.copytree(tables_dir, bundle / "tables", dirs_exist_ok=True)
        (bundle / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_archive = Path(temporary) / archive.name
        with tarfile.open(temporary_archive, "w:gz") as handle:
            handle.add(bundle, arcname=bundle.name, recursive=True)
        checksum = sha256_file(temporary_archive)
        temporary_sidecar = Path(temporary) / sidecar.name
        temporary_sidecar.write_text(f"{checksum}  {archive.name}\n")
        # Publish only complete files. Publishing the sidecar first makes an
        # interrupted attempt retryable because the final archive remains absent.
        os.replace(temporary_sidecar, sidecar)
        os.replace(temporary_archive, archive)
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum}, indent=2))


if __name__ == "__main__":
    main()
