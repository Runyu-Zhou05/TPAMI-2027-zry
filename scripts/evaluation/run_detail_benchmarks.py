#!/usr/bin/env python3
"""Evaluate MMBench and SEED on one immutable 40% detail split."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import prepare_detail_splits as detail_splits
from prepare_evaluation_data import DATASETS, digest
from run_six_benchmarks import (
    VLMEVAL_COMMIT,
    evaluator_fingerprint,
    model_fingerprint,
    require_executable,
    require_prediction_coverage,
    require_scores,
    sha256_bytes,
    sha256_file,
)


DETAIL_DATASETS = {
    "MMBench_DEV_EN_V11": "MMBench_DEV_EN_V11_FVE_HELDOUT",
    "SEEDBench_IMG": "SEEDBench_IMG_FVE_HELDOUT",
}
DETAIL_CONFIG = {
    custom: {"class": "CustomMCQDataset", "dataset": custom}
    for custom in DETAIL_DATASETS.values()
}


def split_fingerprint(split_manifest_path: Path) -> tuple[dict, dict]:
    manifest = json.loads(split_manifest_path.read_text())
    split_id = manifest.get("split_id")
    protocol = manifest.get("protocol", {})
    if not isinstance(split_id, str) or len(split_id) != 64:
        raise SystemExit(f"Malformed detail split ID: {split_manifest_path}")
    if protocol.get("heldout_fraction") != 0.4 or protocol.get("schema_version") != 1:
        raise SystemExit(f"Unsupported detail split protocol: {split_manifest_path}")
    if manifest.get("generator_script_sha256") != sha256_file(Path(detail_splits.__file__)):
        raise SystemExit(f"Detail split generator identity mismatch: {split_manifest_path}")
    import pandas as pd

    data = {}
    partition_identity = {}
    expected_root = (split_manifest_path.parent / "heldout").resolve()
    for name, custom_name in DETAIL_DATASETS.items():
        entry = manifest.get("datasets", {}).get(name, {})
        path = Path(entry.get("heldout_tsv", "")).expanduser().resolve()
        expected_path = expected_root / f"{custom_name}.tsv"
        if path != expected_path or not path.is_file():
            raise SystemExit(f"Missing canonical held-out data for {name}: {expected_path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != entry.get("heldout_tsv_sha256") or path.stat().st_size != entry.get("heldout_tsv_bytes"):
            raise SystemExit(f"Held-out data checksum mismatch for {name}: {path}")
        source = entry.get("source", {})
        source_path = Path(source.get("path", "")).expanduser().resolve()
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source.get("sha256")
            or source_path.stat().st_size != source.get("bytes")
            or digest(source_path, "md5") != DATASETS[name]["md5"]
            or source.get("md5") != DATASETS[name]["md5"]
        ):
            raise SystemExit(f"Detail split source is missing or changed for {name}: {source_path}")
        source_frame = pd.read_csv(source_path, sep="\t")
        if name == "MMBench_DEV_EN_V11":
            expected_heldout, expected_info = detail_splits.split_mmbench(source_frame, int(protocol["seed"]))
        else:
            expected_heldout, expected_info = detail_splits.split_seed(source_frame, int(protocol["seed"]))
        heldout_frame = pd.read_csv(path, sep="\t")
        actual_indices = [str(value) for value in heldout_frame["index"].tolist()]
        expected_indices = [str(value) for value in expected_heldout["index"].tolist()]
        if actual_indices != expected_indices or entry.get("heldout_indices") != expected_info["heldout_indices"]:
            raise SystemExit(f"Held-out indices do not match the deterministic split for {name}")
        if actual_sha256 != detail_splits.dataframe_tsv_sha256(expected_heldout):
            raise SystemExit(f"Held-out rows differ from the pinned source partition for {name}")
        for key in ("heldout_indices_sha256", "train_indices_sha256"):
            if entry.get(key) != expected_info[key]:
                raise SystemExit(f"Detail split {key} mismatch for {name}")
        if entry.get("heldout_rows") != len(expected_heldout) or entry.get("source_rows") != len(source_frame):
            raise SystemExit(f"Detail split row counts mismatch for {name}")
        partition_identity[name] = {
            "heldout_indices_sha256": expected_info["heldout_indices_sha256"],
            "train_indices_sha256": expected_info["train_indices_sha256"],
        }
        data[name] = {
            "heldout_indices_sha256": entry.get("heldout_indices_sha256"),
            "heldout_rows": entry.get("heldout_rows"),
            "sha256": actual_sha256,
        }
    expected_split_id = sha256_bytes(
        json.dumps(
            {"partition": partition_identity, "protocol": protocol}, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    if split_id != expected_split_id:
        raise SystemExit(f"Detail split ID does not match its exact partition: {split_manifest_path}")
    return manifest, {"data": data, "manifest_sha256": sha256_file(split_manifest_path), "split_id": split_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-role", choices=("rotation",), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse only an identical immutable attempt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    for label, value in (("model-name", args.model_name), ("run-id", args.run_id)):
        if not value or any(character not in allowed for character in value):
            raise SystemExit(f"--{label} contains unsupported characters: {value!r}")
    if "," in args.gpu or not args.gpu.isdigit():
        raise SystemExit("--gpu must be exactly one physical GPU index")

    output_root = args.output_root.expanduser().resolve()
    eval_root = (args.eval_root or output_root / "tools" / "vlmevalkit").expanduser().resolve()
    eval_python = require_executable(eval_root / "env" / "bin" / "python")
    vlmeval_root = eval_root / f"VLMEvalKit-{VLMEVAL_COMMIT}"
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Invalid model path: {model_path}")
    split_manifest_path = args.split_manifest.expanduser().resolve()
    if not split_manifest_path.is_file():
        raise SystemExit(f"Missing detail split manifest: {split_manifest_path}")
    split_manifest, split_provenance = split_fingerprint(split_manifest_path)
    heldout_root = (split_manifest_path.parent / "heldout").resolve()

    config = {
        "model": {
            args.model_name: {
                "class": "Qwen2VLChat",
                "model_path": str(model_path),
                "min_pixels": 1280 * 28 * 28,
                "max_pixels": 16384 * 28 * 28,
                "max_new_tokens": 2048,
                "top_p": 0.001,
                "top_k": 1,
                "temperature": 0.01,
                "repetition_penalty": 1.0,
                "use_custom_prompt": False,
            }
        },
        "data": DETAIL_CONFIG,
    }
    model_provenance = model_fingerprint(model_path)
    if model_provenance.get("artifact_type") != "fve_export":
        raise SystemExit("Rotation detail evaluation requires an exported FVE checkpoint manifest")
    package_freeze = subprocess.run(
        [str(eval_python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    provenance = {
        "config": config,
        "evaluator": evaluator_fingerprint(vlmeval_root),
        "model": model_provenance,
        "model_role": args.model_role,
        "package_freeze": package_freeze,
        "split": split_provenance,
    }
    fingerprint = sha256_bytes(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode())
    attempt_name = f"{args.model_name}-{fingerprint[:16]}"
    work_dir = output_root / "evaluation" / "detail" / args.run_id / attempt_name
    config_path = work_dir / "vlmeval_config.json"
    preflight_path = work_dir / "attempt_preflight.json"
    manifest_path = work_dir / "detail_benchmark_run_manifest.json"
    command = [
        str(eval_python),
        str(vlmeval_root / "run.py"),
        "--config",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--judge",
        "exact_matching",
    ]
    if args.resume:
        command.append("--reuse")
    if args.dry_run:
        print(json.dumps({"attempt": str(work_dir), "command": command, "fingerprint": fingerprint}, indent=2))
        return

    if work_dir.exists() and any(work_dir.iterdir()):
        if not args.resume or not preflight_path.is_file():
            raise SystemExit(f"Refusing to reuse nonempty detail attempt: {work_dir}")
        existing = json.loads(preflight_path.read_text())
        if existing != {"fingerprint": fingerprint, "provenance": provenance}:
            raise SystemExit(f"Detail attempt fingerprint mismatch: {work_dir}")
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise SystemExit(f"Detail attempt config changed: {config_path}")
        if manifest_path.is_file():
            completed = json.loads(manifest_path.read_text())
            if completed.get("attempt_fingerprint") != fingerprint:
                raise SystemExit(f"Completed detail fingerprint mismatch: {manifest_path}")
            if set(completed.get("score_artifacts", {})) != set(DETAIL_DATASETS) or set(
                completed.get("prediction_artifacts", {})
            ) != set(DETAIL_DATASETS):
                raise SystemExit(f"Completed detail manifest is incomplete: {manifest_path}")
            for name, artifact in completed.get("score_artifacts", {}).items():
                if sha256_file(Path(artifact)) != completed.get("score_artifact_sha256", {}).get(name):
                    raise SystemExit(f"Completed detail score changed: {artifact}")
            for record in completed.get("prediction_artifacts", {}).values():
                if sha256_file(Path(record["path"])) != record.get("sha256"):
                    raise SystemExit(f"Completed detail prediction changed: {record['path']}")
            print(json.dumps(completed, indent=2, sort_keys=True))
            return
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        preflight_path.write_text(json.dumps({"fingerprint": fingerprint, "provenance": provenance}, indent=2, sort_keys=True) + "\n")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["LMUData"] = str(heldout_root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_path = work_dir / "detail_benchmark.log"
    with log_path.open("a" if args.resume else "x", buffering=1) as log:
        subprocess.run(command, cwd=vlmeval_root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    custom_scores = require_scores(work_dir, args.model_name, DETAIL_CONFIG)
    custom_predictions = require_prediction_coverage(
        eval_python, work_dir, args.model_name, heldout_root, DETAIL_CONFIG
    )
    scores = {logical: custom_scores[custom] for logical, custom in DETAIL_DATASETS.items()}
    predictions = {logical: custom_predictions[custom] for logical, custom in DETAIL_DATASETS.items()}
    manifest = {
        "attempt_fingerprint": fingerprint,
        "benchmarks": list(DETAIL_DATASETS),
        "command": command,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "detail_40_percent_heldout",
        "gpu": args.gpu,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "model_role": args.model_role,
        "package_freeze": package_freeze,
        "provenance": provenance,
        "prediction_artifacts": predictions,
        "run_id": args.run_id,
        "schema_version": 1,
        "score_artifact_sha256": {name: sha256_file(Path(path)) for name, path in scores.items()},
        "score_artifacts": scores,
        "score_input_prediction_sha256": {
            dataset: record["sha256"] for dataset, record in predictions.items()
        },
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": split_provenance["manifest_sha256"],
        "split_id": split_manifest["split_id"],
        "vlmeval_commit": VLMEVAL_COMMIT,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
