#!/usr/bin/env python3
"""Validate complete FVE lineage and publish a curated result archive."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from run_detail_benchmarks import split_fingerprint


ROOT = Path(__file__).resolve().parents[2]
FULL_BENCHMARKS = {
    "MMBench_DEV_EN_V11",
    "SEEDBench_IMG",
    "VStarBench",
    "RealWorldQA",
    "BLINK",
    "MME-RealWorld-Lite",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        return path, json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed {label}: {path}") from exc


def validate_file_record(record: dict, label: str) -> Path:
    path = Path(record.get("path", "")).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing recorded {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
        raise SystemExit(f"Recorded {label} has changed: {path}")
    return path


def validate_score_artifacts(path: Path, manifest: dict, required: set[str]) -> None:
    artifacts = manifest.get("score_artifacts", {})
    hashes = manifest.get("score_artifact_sha256", {})
    if not required.issubset(artifacts):
        raise SystemExit(f"Evaluation manifest lacks scores {sorted(required - set(artifacts))}: {path}")
    for name in required:
        artifact = Path(artifacts[name]).expanduser().resolve()
        if not artifact.is_file() or sha256_file(artifact) != hashes.get(name):
            raise SystemExit(f"Evaluation score artifact is missing or changed: {artifact}")


def runtime_signature(manifest: dict) -> dict:
    provenance = manifest.get("provenance", {})
    models = list(provenance.get("config", {}).get("model", {}).values())
    if len(models) != 1:
        raise SystemExit("Evaluation provenance must contain exactly one model config")
    generation = dict(models[0])
    generation.pop("model_path", None)
    return {"evaluator": provenance.get("evaluator"), "generation": generation}


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


def inventory_tree(root: Path) -> list[dict]:
    if not root.exists():
        return []
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
        )
    return files


def copy_curated_training(training_dir: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    patterns = (
        "*.log",
        "summary*.json",
        "*manifest.json",
        "checkpoint_complete.json",
        "checkpoint_stage.*.json",
        "latest_checkpointed_iteration.txt",
    )
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


def validate_evaluation(path: Path, manifest: dict, role: str, run_id: str) -> None:
    if (
        manifest.get("schema_version") != 2
        or manifest.get("evaluation_role") != "full_six_benchmarks"
        or manifest.get("model_role") != role
        or manifest.get("run_id") != run_id
    ):
        raise SystemExit(f"Invalid {role} full-evaluation identity: {path}")
    if set(manifest.get("benchmarks", [])) != FULL_BENCHMARKS:
        raise SystemExit(f"Incomplete benchmark set in {path}")
    validate_score_artifacts(path, manifest, FULL_BENCHMARKS)
    predictions = manifest.get("prediction_artifacts", {})
    if set(predictions) != FULL_BENCHMARKS:
        raise SystemExit(f"Incomplete prediction evidence in {path}")
    for name, record in predictions.items():
        prediction = Path(record.get("path", "")).expanduser().resolve()
        if (
            not prediction.is_file()
            or sha256_file(prediction) != record.get("sha256")
            or int(record.get("rows", 0)) <= 0
            or not record.get("sorted_index_sha256")
        ):
            raise SystemExit(f"Prediction evidence is missing or changed for {name}: {prediction}")


def validate_detail_evaluation(path: Path, manifest: dict, role: str, run_id: str, full: dict) -> None:
    required = {"MMBench_DEV_EN_V11", "SEEDBench_IMG"}
    if (
        manifest.get("schema_version") != 1
        or manifest.get("evaluation_role") != "detail_40_percent_heldout"
        or manifest.get("model_role") != role
        or manifest.get("run_id") != run_id
    ):
        raise SystemExit(f"Invalid {role} detail-evaluation identity: {path}")
    validate_score_artifacts(path, manifest, required)
    predictions = manifest.get("prediction_artifacts", {})
    if set(predictions) != required:
        raise SystemExit(f"Incomplete detail prediction evidence in {path}")
    for name, record in predictions.items():
        prediction = Path(record.get("path", "")).expanduser().resolve()
        if not prediction.is_file() or sha256_file(prediction) != record.get("sha256"):
            raise SystemExit(f"Detail prediction evidence changed for {name}: {prediction}")
    if manifest.get("provenance", {}).get("model") != full.get("provenance", {}).get("model"):
        raise SystemExit(f"{role} detail evaluation used a different model from its full evaluation")
    if runtime_signature(manifest) != runtime_signature(full):
        raise SystemExit(f"{role} detail evaluation used a different evaluator or generation protocol")


def validate_tables(tables_dir: Path, run_id: str, expected_inputs: dict[str, Path]) -> dict:
    manifest_path, manifest = load_json(tables_dir / "paper_tables.json", "paper table manifest")
    if manifest.get("schema_version") != 2 or manifest.get("run_id") != run_id:
        raise SystemExit(f"Paper table identity mismatch: {manifest_path}")
    inputs = manifest.get("inputs", {})
    for name, expected_path in expected_inputs.items():
        record = inputs.get(name)
        if not isinstance(record, dict):
            raise SystemExit(f"Paper tables lack required input {name}")
        actual_path = Path(record.get("path", "")).expanduser().resolve()
        if actual_path != expected_path or record.get("sha256") != sha256_file(expected_path):
            raise SystemExit(f"Paper table input mismatch for {name}")
    for name, record in inputs.items():
        if record is None:
            continue
        input_path = Path(record.get("path", "")).expanduser().resolve()
        if not input_path.is_file() or sha256_file(input_path) != record.get("sha256"):
            raise SystemExit(f"Paper table input changed for {name}: {input_path}")
    for name, expected_hash in manifest.get("artifact_sha256", {}).items():
        artifact = tables_dir / name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise SystemExit(f"Paper table artifact changed: {artifact}")
    return manifest


def validate_lineage(args) -> dict:
    run_id = args.run_id
    queue_path, queue = load_json(args.queue_manifest, "queue completion manifest")
    post_path, post = load_json(args.post_training_manifest, "post-training manifest")
    data_path, data = load_json(args.rotation_data_manifest, "Rotation data manifest")
    base_path, base = load_json(args.base_eval_manifest, "base evaluation manifest")
    rotation_path, rotation = load_json(args.rotation_eval_manifest, "Rotation evaluation manifest")
    training_dir = args.training_run_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()

    if not training_dir.is_dir() or not tables_dir.is_dir():
        raise SystemExit("Training run and table paths must be directories")
    if queue.get("status") != "complete" or queue.get("schema_version") != 1 or queue.get("run_id") != run_id:
        raise SystemExit(f"Queue did not complete this run: {queue_path}")
    if queue.get("invocation_provenance", {}).get("environment", {}).get("validated") is not True:
        raise SystemExit(f"Queue lacks a validated environment record: {queue_path}")
    stages = queue.get("stages", [])
    if [stage.get("mode") for stage in stages] != ["quick", "quick-resume", "production-smoke", "train"]:
        raise SystemExit(f"Queue stage sequence is incomplete: {queue_path}")
    for stage in stages:
        if stage.get("task") != "rotation":
            raise SystemExit(f"Queue contains a non-Rotation stage: {queue_path}")
        validate_file_record(stage.get("checkpoint", {}).get("completion_manifest", {}), "stage checkpoint manifest")
        validate_file_record(stage.get("checkpoint", {}).get("tracker", {}), "stage checkpoint tracker")
        validate_file_record(stage.get("reward_summary", {}), "stage reward summary")
        stage_run_dir = Path(stage.get("run_dir", "")).expanduser().resolve()
        if training_dir not in stage_run_dir.parents:
            raise SystemExit("Queue stage is outside --training-run-dir")
    final_stage = stages[-1]
    final_checkpoint = final_stage.get("checkpoint", {})
    validate_file_record(final_checkpoint.get("completion_manifest", {}), "final checkpoint manifest")
    validate_file_record(final_stage.get("reward_summary", {}), "final reward summary")

    data_record = queue.get("invocation_provenance", {}).get("data_manifest", {})
    if validate_file_record(data_record, "queued Rotation data manifest") != data_path:
        raise SystemExit("Queue used a different Rotation data manifest")
    if data.get("complete") is not True or data.get("split_mapping") != {"test": "train.parquet", "validation": "test.parquet"}:
        raise SystemExit(f"Rotation data manifest has the wrong complete split mapping: {data_path}")
    for name in ("train.parquet", "test.parquet"):
        if name not in data.get("files", {}):
            raise SystemExit(f"Rotation data manifest lacks {name}: {data_path}")

    if (
        post.get("schema_version") != 2
        or post.get("model_role") != "rotation"
        or post.get("run_id") != run_id
    ):
        raise SystemExit(f"Invalid post-training identity: {post_path}")
    checkpoint_hash = final_checkpoint.get("completion_manifest", {}).get("sha256")
    if post.get("provenance", {}).get("checkpoint_manifest_sha256") != checkpoint_hash:
        raise SystemExit("Post-training smoke did not use the queue's final committed checkpoint")
    export_manifest = Path(post.get("export", {}).get("exported_model", "")) / "fve_export_manifest.json"
    if not export_manifest.is_file() or sha256_file(export_manifest) != post.get("export_manifest_sha256"):
        raise SystemExit("Post-training export manifest is missing or changed")
    export_payload = json.loads(export_manifest.read_text())
    export_hashes = export_payload.get("audit", {}).get("safetensor_sha256", {})
    if not isinstance(export_hashes, dict) or not export_hashes:
        raise SystemExit("Post-training export manifest contains no weight hashes")
    for name, expected_hash in export_hashes.items():
        shard = export_manifest.parent / name
        if not shard.is_file() or sha256_file(shard) != expected_hash:
            raise SystemExit(f"Post-training exported weight is missing or changed: {shard}")
    export_support_hashes = export_payload.get("audit", {}).get("support_file_sha256", {})
    for name, expected_hash in export_support_hashes.items():
        support = export_manifest.parent / name
        if not support.is_file() or sha256_file(support) != expected_hash:
            raise SystemExit(f"Post-training exported support file is missing or changed: {support}")
    smoke_manifest_path = post_path.parent / "mmbench_smoke" / "smoke_manifest.json"
    if (
        not smoke_manifest_path.is_file()
        or sha256_file(smoke_manifest_path) != post.get("mmbench_smoke_manifest_sha256")
        or json.loads(smoke_manifest_path.read_text()) != post.get("mmbench_smoke")
    ):
        raise SystemExit("Post-training MMBench smoke manifest is missing or changed")
    smoke = post["mmbench_smoke"]
    expected_fields = {"Logical", "Relation", "Attribute", "Coarse", "Cross-Inst.", "Single-Inst.", "Average"}
    required_smoke_artifacts = {
        "mmbench_v11_smoke_official_scores.csv",
        "mmbench_v11_smoke_predictions.xlsx",
        "mmbench_v11_smoke_subset.tsv",
        "mmbench_v11_smoke_table2.json",
        "smoke_manifest.json",
    }
    if (
        int(smoke.get("sample_count", 0)) != 24
        or set(smoke.get("table2_smoke_percent", {})) != expected_fields
        or any(value is None for value in smoke.get("table2_smoke_percent", {}).values())
        or Path(smoke.get("model_path", "")).expanduser().resolve() != export_manifest.parent.resolve()
        or not required_smoke_artifacts.issubset(post.get("smoke_artifacts", {}))
    ):
        raise SystemExit("Post-training MMBench smoke evidence is incomplete or bound to the wrong model")
    table_projection = validate_file_record(
        post["smoke_artifacts"]["mmbench_v11_smoke_table2.json"], "post-training smoke Table II projection"
    )
    if json.loads(table_projection.read_text()) != smoke["table2_smoke_percent"]:
        raise SystemExit("Post-training smoke Table II projection disagrees with its manifest")
    for name, record in post.get("smoke_artifacts", {}).items():
        validate_file_record(record, f"post-training smoke artifact {name}")
    if not post.get("smoke_artifacts"):
        raise SystemExit("Post-training manifest contains no smoke artifact inventory")
    validate_file_record(post.get("smoke_log", {}), "post-training smoke log")

    validate_evaluation(base_path, base, "base", run_id)
    validate_evaluation(rotation_path, rotation, "rotation", run_id)
    rotation_export_hash = rotation.get("provenance", {}).get("model", {}).get("export_manifest_sha256")
    if rotation_export_hash != post.get("export_manifest_sha256"):
        raise SystemExit("Rotation full evaluation did not use the post-training-smoke export")
    if Path(rotation.get("model_path", "")).expanduser().resolve() != export_manifest.parent.resolve():
        raise SystemExit("Rotation full evaluation used a different exported model path")
    if rotation.get("provenance", {}).get("model", {}).get("safetensor_sha256") != export_hashes:
        raise SystemExit("Rotation full evaluation used different exported model bytes")
    if rotation.get("provenance", {}).get("model", {}).get("support_file_sha256") != export_support_hashes:
        raise SystemExit("Rotation full evaluation used different exported config/processor/tokenizer bytes")
    for key in ("data", "data_manifest_sha256", "evaluator"):
        if base.get("provenance", {}).get(key) != rotation.get("provenance", {}).get(key):
            raise SystemExit(f"Base and Rotation full evaluations differ in {key}")

    table_manifest = validate_tables(
        tables_dir,
        run_id,
        {"base_full": base_path, "rotation_full": rotation_path},
    )
    lineage = {
        "base_evaluation_manifest": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "post_training_manifest": {"path": str(post_path), "sha256": sha256_file(post_path)},
        "queue_manifest": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "rotation_data_manifest": {"path": str(data_path), "sha256": sha256_file(data_path)},
        "rotation_evaluation_manifest": {"path": str(rotation_path), "sha256": sha256_file(rotation_path)},
        "table_manifest": {"path": str(tables_dir / "paper_tables.json"), "sha256": sha256_file(tables_dir / "paper_tables.json")},
        "table_protocol": table_manifest.get("detail_protocol"),
    }
    detail_records = {
        name: table_manifest.get("inputs", {}).get(name) for name in ("base_detail", "rotation_detail")
    }
    if (detail_records["base_detail"] is None) != (detail_records["rotation_detail"] is None):
        raise SystemExit("Final tables contain an unpaired detail evaluation")
    if detail_records["base_detail"] is None or table_manifest.get("detail_protocol") != "deterministic 40% held-out split":
        raise SystemExit("Final paper package requires the paired deterministic 40% detail evaluations")
    if detail_records["base_detail"] is not None:
        base_detail_path, base_detail = load_json(
            Path(detail_records["base_detail"]["path"]), "base detail evaluation manifest"
        )
        rotation_detail_path, rotation_detail = load_json(
            Path(detail_records["rotation_detail"]["path"]), "Rotation detail evaluation manifest"
        )
        validate_detail_evaluation(base_detail_path, base_detail, "base", run_id, base)
        validate_detail_evaluation(rotation_detail_path, rotation_detail, "rotation", run_id, rotation)
        if (
            base_detail.get("split_id") != rotation_detail.get("split_id")
            or base_detail.get("provenance", {}).get("split")
            != rotation_detail.get("provenance", {}).get("split")
        ):
            raise SystemExit("Base and Rotation detail evaluations use different held-out rows")
        split_path = Path(base_detail.get("split_manifest", "")).expanduser().resolve()
        split_hash = base_detail.get("split_manifest_sha256")
        if (
            not split_path.is_file()
            or sha256_file(split_path) != split_hash
            or rotation_detail.get("split_manifest_sha256") != split_hash
            or rotation_detail.get("split_manifest") != str(split_path)
        ):
            raise SystemExit("Detail split manifest is missing, changed, or inconsistent")
        verified_split, _ = split_fingerprint(split_path)
        if verified_split.get("split_id") != base_detail.get("split_id"):
            raise SystemExit("Detail split manifest fails deterministic partition verification")
        lineage["base_detail_manifest"] = {
            "path": str(base_detail_path),
            "sha256": sha256_file(base_detail_path),
        }
        lineage["rotation_detail_manifest"] = {
            "path": str(rotation_detail_path),
            "sha256": sha256_file(rotation_detail_path),
        }
        lineage["detail_split_manifest"] = {"path": str(split_path), "sha256": split_hash}
    return lineage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path, required=True)
    parser.add_argument("--post-training-manifest", type=Path, required=True)
    parser.add_argument("--rotation-data-manifest", type=Path, required=True)
    parser.add_argument("--base-eval-manifest", type=Path, required=True)
    parser.add_argument("--rotation-eval-manifest", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not args.run_id or any(character not in allowed for character in args.run_id):
        raise SystemExit("--run-id contains unsupported characters")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = (output_dir / f".fve_{args.run_id}.package.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"Another packager is already publishing run {args.run_id}") from exc
    archive = output_dir / f"fve_{args.run_id}_results.tar.gz"
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if archive.exists():
        raise SystemExit(f"Refusing to overwrite result archive: {archive}")

    lineage = validate_lineage(args)
    training_dir = args.training_run_dir.expanduser().resolve()
    base_eval_dir = args.base_eval_manifest.expanduser().resolve().parent
    rotation_eval_dir = args.rotation_eval_manifest.expanduser().resolve().parent
    post_dir = args.post_training_manifest.expanduser().resolve().parent
    tables_dir = args.tables_dir.expanduser().resolve()
    detail_dirs = {
        role: Path(lineage[key]["path"]).parent
        for role, key in (("base", "base_detail_manifest"), ("rotation", "rotation_detail_manifest"))
        if key in lineage
    }
    manifest = {
        "base_evaluation": inventory_tree(base_eval_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "lineage": lineage,
        "detail_evaluations": {role: inventory_tree(path) for role, path in detail_dirs.items()},
        "post_training": inventory_tree(post_dir),
        "rotation_evaluation": inventory_tree(rotation_eval_dir),
        "run_id": args.run_id,
        "schema_version": 2,
        "tables": inventory_tree(tables_dir),
        "training": inventory_tree(training_dir),
    }
    with tempfile.TemporaryDirectory(prefix=f".fve-{args.run_id}-bundle-", dir=output_dir) as temporary:
        bundle = Path(temporary) / f"fve_{args.run_id}_results"
        bundle.mkdir()
        copy_curated_training(training_dir, bundle / "training")
        shutil.copytree(base_eval_dir, bundle / "evaluation" / "base", dirs_exist_ok=True)
        shutil.copytree(rotation_eval_dir, bundle / "evaluation" / "rotation", dirs_exist_ok=True)
        for role, path in detail_dirs.items():
            shutil.copytree(path, bundle / "evaluation" / "detail" / role, dirs_exist_ok=True)
        shutil.copytree(post_dir, bundle / "post_training", dirs_exist_ok=True)
        shutil.copytree(tables_dir, bundle / "tables", dirs_exist_ok=True)
        workflow = bundle / "workflow"
        workflow.mkdir()
        for label, source in (
            ("queue.complete.json", args.queue_manifest),
            ("rotation_data_manifest.json", args.rotation_data_manifest),
        ):
            shutil.copy2(Path(source).expanduser().resolve(), workflow / label)
        if "detail_split_manifest" in lineage:
            shutil.copy2(Path(lineage["detail_split_manifest"]["path"]), workflow / "detail_split_manifest.json")
        (bundle / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_archive = Path(temporary) / archive.name
        with tarfile.open(temporary_archive, "w:gz") as handle:
            handle.add(bundle, arcname=bundle.name, recursive=True)
        checksum = sha256_file(temporary_archive)
        temporary_sidecar = Path(temporary) / sidecar.name
        temporary_sidecar.write_text(f"{checksum}  {archive.name}\n")
        os.replace(temporary_sidecar, sidecar)
        os.replace(temporary_archive, archive)
    lock.close()
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum}, indent=2))


if __name__ == "__main__":
    main()
