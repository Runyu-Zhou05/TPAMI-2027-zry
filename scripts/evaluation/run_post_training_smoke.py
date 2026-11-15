#!/usr/bin/env python3
"""Export an FVE checkpoint and run the validated MMBench smoke in one command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from export_fve_checkpoint import resolve_actor_dir, sha256_file


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "scripts/evaluation/export_fve_checkpoint.py"
MMBENCH_SMOKE = ROOT / "scripts/evaluation/vlmeval_mmbench_smoke.py"
DEFAULT_OUTPUT_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv")
EXPECTED_TABLE_FIELDS = {"Logical", "Relation", "Attribute", "Coarse", "Cross-Inst.", "Single-Inst.", "Average"}
REQUIRED_SMOKE_ARTIFACTS = {
    "mmbench_v11_smoke_official_scores.csv",
    "mmbench_v11_smoke_predictions.xlsx",
    "mmbench_v11_smoke_subset.tsv",
    "mmbench_v11_smoke_table2.json",
    "smoke_manifest.json",
}


def executable(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Executable does not exist: {path}")
    return str(path.resolve())


def file_record(path: Path) -> dict:
    path = path.expanduser().resolve()
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": sha256_file(path)}


def validate_file_record(record: dict, label: str) -> Path:
    path = Path(record.get("path", "")).expanduser().resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise SystemExit(f"Post-training {label} is missing or changed: {path}")
    return path


def validate_export(export_manifest: dict, expected_manifest_sha256: str) -> Path:
    model_dir = Path(export_manifest.get("exported_model", "")).expanduser().resolve()
    manifest_path = model_dir / "fve_export_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise SystemExit(f"Post-training export manifest is missing or changed: {manifest_path}")
    hashes = export_manifest.get("audit", {}).get("safetensor_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        raise SystemExit(f"Post-training export has no weight hashes: {manifest_path}")
    for name, expected_hash in hashes.items():
        shard = model_dir / name
        if not shard.is_file() or sha256_file(shard) != expected_hash:
            raise SystemExit(f"Post-training exported weight is missing or changed: {shard}")
    support_hashes = export_manifest.get("audit", {}).get("support_file_sha256", {})
    for name, expected_hash in support_hashes.items():
        support = model_dir / name
        if not support.is_file() or sha256_file(support) != expected_hash:
            raise SystemExit(f"Post-training exported support file is missing or changed: {support}")
    return model_dir


def validate_completed_attempt(manifest: dict, fingerprint: str, run_id: str) -> None:
    if (
        manifest.get("schema_version") != 2
        or manifest.get("attempt_fingerprint") != fingerprint
        or manifest.get("run_id") != run_id
        or manifest.get("model_role") != "rotation"
    ):
        raise SystemExit("Post-training completion manifest identity mismatch")
    exported_model = validate_export(manifest.get("export", {}), manifest.get("export_manifest_sha256"))
    smoke_manifest_path = validate_file_record(
        manifest.get("smoke_artifacts", {}).get("smoke_manifest.json", {}), "smoke manifest"
    )
    smoke = json.loads(smoke_manifest_path.read_text())
    artifacts = manifest.get("smoke_artifacts", {})
    if (
        smoke != manifest.get("mmbench_smoke")
        or Path(smoke.get("model_path", "")).expanduser().resolve() != exported_model
        or int(smoke.get("sample_count", 0)) != 24
        or set(smoke.get("table2_smoke_percent", {})) != EXPECTED_TABLE_FIELDS
        or any(value is None for value in smoke.get("table2_smoke_percent", {}).values())
        or not REQUIRED_SMOKE_ARTIFACTS.issubset(artifacts)
    ):
        raise SystemExit(f"Post-training smoke evidence is incomplete or inconsistent: {smoke_manifest_path}")
    table_path = validate_file_record(artifacts["mmbench_v11_smoke_table2.json"], "smoke Table II projection")
    if json.loads(table_path.read_text()) != smoke["table2_smoke_percent"]:
        raise SystemExit(f"Post-training smoke Table II projection changed: {table_path}")
    for name, record in artifacts.items():
        validate_file_record(record, f"smoke artifact {name}")
    validate_file_record(manifest.get("smoke_log", {}), "smoke log")


def fingerprint_payload(args, checkpoint: Path, training_python: str, eval_python: str) -> dict:
    actor_dir = resolve_actor_dir(checkpoint)
    completion_manifest = actor_dir.parent / "checkpoint_complete.json"
    completion = json.loads(completion_manifest.read_text())
    if completion.get("schema_version") != 2 or completion.get("run_id") != args.run_id:
        raise SystemExit(
            f"Committed checkpoint belongs to run {completion.get('run_id')!r}, not {args.run_id!r}"
        )
    payload = {
        "checkpoint_actor": str(actor_dir),
        "checkpoint_manifest_sha256": sha256_file(completion_manifest),
        "eval_python": eval_python,
        "run_id": args.run_id,
        "scripts": {
            path.name: sha256_file(path) for path in (EXPORTER, MMBENCH_SMOKE, Path(__file__).resolve())
        },
        "source_tsv": None,
        "training_python": training_python,
    }
    if args.source_tsv is not None:
        source = args.source_tsv.expanduser().resolve()
        if not source.is_file():
            raise SystemExit(f"Missing MMBench source TSV: {source}")
        payload["source_tsv"] = {"path": str(source), "sha256": sha256_file(source)}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True, help="One physical GPU index for MMBench smoke")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("FVE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--training-python", default=os.environ.get("FVE_PYTHON", sys.executable))
    parser.add_argument("--eval-python", default=None)
    parser.add_argument("--source-tsv", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse only an identical completed attempt")
    args = parser.parse_args()

    if not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in args.run_id):
        raise SystemExit("--run-id may contain only letters, digits, '.', '_' and '-'")
    if "," in args.gpu or not args.gpu.strip().isdigit():
        raise SystemExit("--gpu must be exactly one physical GPU index")

    output_root = args.output_root.expanduser().resolve()
    training_python = executable(args.training_python)
    eval_python = executable(
        args.eval_python or output_root / "tools" / "vlmevalkit" / "env" / "bin" / "python"
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    provenance = fingerprint_payload(args, checkpoint, training_python, eval_python)
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result_dir = output_root / "evaluation" / "post_training" / args.run_id / fingerprint[:16]
    manifest_path = result_dir / "post_training_manifest.json"
    if result_dir.exists() and any(result_dir.iterdir()):
        if not args.resume or not manifest_path.is_file():
            raise SystemExit(f"Refusing to reuse nonempty post-training attempt: {result_dir}")
        existing = json.loads(manifest_path.read_text())
        validate_completed_attempt(existing, fingerprint, args.run_id)
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "attempt_preflight.json").write_text(
        json.dumps({"fingerprint": fingerprint, "provenance": provenance}, indent=2, sort_keys=True) + "\n"
    )
    export_result = result_dir / "export_result.json"

    subprocess.run(
        [
            training_python,
            str(EXPORTER),
            "--checkpoint",
            str(checkpoint),
            "--python",
            training_python,
            "--result-json",
            str(export_result),
        ],
        cwd=ROOT,
        check=True,
    )
    export_manifest = json.loads(export_result.read_text())
    exported_model = Path(export_manifest["exported_model"])

    smoke_dir = result_dir / "mmbench_smoke"
    smoke_command = [
        eval_python,
        str(MMBENCH_SMOKE),
        "--model-path",
        str(exported_model),
        "--lmu-data-root",
        str(output_root / "inputs" / "vlmeval"),
        "--output-dir",
        str(smoke_dir),
    ]
    if args.source_tsv is not None:
        smoke_command.extend(["--source-tsv", str(args.source_tsv.expanduser().resolve())])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with (result_dir / "mmbench_smoke.log").open("w", buffering=1) as log:
        subprocess.run(
            smoke_command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    smoke_manifest_path = smoke_dir / "smoke_manifest.json"
    if not smoke_manifest_path.is_file():
        raise SystemExit(f"MMBench smoke did not produce its manifest: {smoke_manifest_path}")
    smoke_manifest = json.loads(smoke_manifest_path.read_text())
    smoke_artifacts = {
        str(path.relative_to(smoke_dir)): file_record(path)
        for path in sorted(item for item in smoke_dir.rglob("*") if item.is_file())
    }
    manifest = {
        "attempt_fingerprint": fingerprint,
        "checkpoint": str(checkpoint),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "export": export_manifest,
        "export_manifest_sha256": sha256_file(Path(export_manifest["exported_model"]) / "fve_export_manifest.json"),
        "gpu": args.gpu.strip(),
        "mmbench_smoke": smoke_manifest,
        "mmbench_smoke_manifest_sha256": sha256_file(smoke_manifest_path),
        "model_role": "rotation",
        "provenance": provenance,
        "run_id": args.run_id,
        "schema_version": 2,
        "smoke_artifacts": smoke_artifacts,
        "smoke_log": file_record(result_dir / "mmbench_smoke.log"),
    }
    validate_completed_attempt(manifest, fingerprint, args.run_id)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
