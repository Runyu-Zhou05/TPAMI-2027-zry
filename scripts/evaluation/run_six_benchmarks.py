#!/usr/bin/env python3
"""Run immutable, provenance-bound six-benchmark VLMEvalKit attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare_evaluation_data import DATASETS, digest


VLMEVAL_COMMIT = "0a11917eede83742389b03d7f11fc252aabdf6ef"
BENCHMARKS = {
    "MMBench_DEV_EN_V11": {"class": "ImageMCQDataset", "dataset": "MMBench_DEV_EN_V11"},
    "SEEDBench_IMG": {"class": "ImageMCQDataset", "dataset": "SEEDBench_IMG"},
    "VStarBench": {"class": "ImageMCQDataset", "dataset": "VStarBench"},
    "RealWorldQA": {"class": "ImageMCQDataset", "dataset": "RealWorldQA"},
    "BLINK": {"class": "ImageMCQDataset", "dataset": "BLINK"},
    "MME-RealWorld-Lite": {"class": "MMERealWorld", "dataset": "MME-RealWorld-Lite"},
}
MODEL_METADATA_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git_blob_sha1(path: Path) -> str:
    result = hashlib.sha1()
    result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_snapshot_metadata(model_path: Path, names: list[str]) -> str:
    revision = None
    for name in names:
        path = model_path / name
        metadata = model_path / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        if not path.is_file() or not metadata.is_file():
            raise SystemExit(f"Base model snapshot file or metadata is missing: {path}")
        lines = metadata.read_text().splitlines()
        if len(lines) < 2:
            raise SystemExit(f"Malformed Hugging Face snapshot metadata: {metadata}")
        current_revision = lines[0].strip()
        if revision is None:
            revision = current_revision
        elif current_revision != revision:
            raise SystemExit(f"Base model snapshot mixes revisions: {metadata}")
        etag = lines[1].strip().removeprefix("W/").strip('"')
        actual_sha256 = sha256_file(path)
        if etag != actual_sha256 and etag != git_blob_sha1(path):
            raise SystemExit(f"Base model snapshot file changed after download: {path}")
    return revision


def require_executable(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Executable is unavailable: {path}")
    return path


def model_fingerprint(model_path: Path) -> dict:
    metadata_hashes = {
        name: sha256_file(model_path / name) for name in MODEL_METADATA_FILES if (model_path / name).is_file()
    }
    if "config.json" not in metadata_hashes:
        raise SystemExit(f"Model lacks config.json: {model_path}")
    export_manifest = model_path / "fve_export_manifest.json"
    if export_manifest.is_file():
        payload = json.loads(export_manifest.read_text())
        audit = payload.get("audit", {})
        hashes = audit.get("safetensor_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise SystemExit(f"Export manifest lacks safetensor hashes: {export_manifest}")
        for name, expected_sha256 in hashes.items():
            shard = model_path / name
            if not shard.is_file() or sha256_file(shard) != expected_sha256:
                raise SystemExit(f"Exported model shard is missing or changed: {shard}")
        support_hashes = audit.get("support_file_sha256")
        if not isinstance(support_hashes, dict):
            raise SystemExit(f"Export manifest lacks support-file hashes: {export_manifest}")
        current_support = {
            str(path.relative_to(model_path)): sha256_file(path)
            for path in sorted(item for item in model_path.rglob("*") if item.is_file())
            if path.suffix != ".safetensors" and path.name != "fve_export_manifest.json"
        }
        if current_support != support_hashes:
            raise SystemExit(f"Exported model config/processor/tokenizer files changed: {model_path}")
        return {
            "artifact_type": "fve_export",
            "model_metadata_sha256": metadata_hashes,
            "export_manifest": str(export_manifest),
            "export_manifest_sha256": sha256_file(export_manifest),
            "safetensor_sha256": hashes,
            "support_file_sha256": support_hashes,
        }

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = sorted(set(index.get("weight_map", {}).values()))
    else:
        names = [path.name for path in sorted(model_path.glob("*.safetensors"))]
    if not names:
        raise SystemExit(f"No safetensors found under {model_path}")
    hashes = {}
    for name in names:
        path = model_path / name
        if not path.is_file():
            raise SystemExit(f"Missing model shard: {path}")
        hashes[name] = sha256_file(path)
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    metadata_names = sorted(
        str(path.relative_to(metadata_root))[: -len(".metadata")]
        for path in metadata_root.rglob("*.metadata")
    )
    required_names = {"config.json", *names}
    if index_path.is_file():
        required_names.add(index_path.name)
    if not required_names.issubset(metadata_names):
        raise SystemExit(f"Base model snapshot metadata is incomplete: {sorted(required_names - set(metadata_names))}")
    revision = validate_snapshot_metadata(model_path, metadata_names)
    return {
        "artifact_type": "huggingface_snapshot",
        "model_metadata_sha256": metadata_hashes,
        "huggingface_revision": revision,
        "safetensor_sha256": hashes,
    }


def evaluator_fingerprint(vlmeval_root: Path) -> dict:
    marker = vlmeval_root / ".pinned-commit"
    if not marker.is_file() or marker.read_text().strip() != VLMEVAL_COMMIT:
        raise SystemExit(f"Pinned VLMEvalKit is unavailable under {vlmeval_root}")
    files = [vlmeval_root / "run.py", *sorted((vlmeval_root / "vlmeval").rglob("*.py"))]
    for path in files:
        if not path.is_file():
            raise SystemExit(f"Pinned evaluator file is missing: {path}")
    return {"commit": VLMEVAL_COMMIT, "files": {str(path.relative_to(vlmeval_root)): sha256_file(path) for path in files}}


def dataset_fingerprint(lmu_root: Path, manifest: dict) -> dict:
    result = {}
    for name in BENCHMARKS:
        info = manifest.get("benchmarks", {}).get(name)
        if not isinstance(info, dict):
            raise SystemExit(f"Evaluation data manifest is missing {name}")
        canonical = (lmu_root / f"{name}.tsv").resolve()
        recorded = Path(info.get("path", "")).expanduser().resolve()
        if canonical != recorded or not canonical.is_file():
            raise SystemExit(f"Non-canonical evaluation dataset path for {name}: {recorded}")
        actual_md5 = digest(canonical, "md5")
        actual_sha256 = digest(canonical, "sha256")
        if actual_md5 != DATASETS[name]["md5"] or actual_md5 != info.get("md5"):
            raise SystemExit(f"Evaluation dataset MD5 mismatch for {name}")
        if actual_sha256 != info.get("sha256"):
            raise SystemExit(f"Evaluation dataset SHA256 mismatch for {name}")
        result[name] = {"bytes": canonical.stat().st_size, "md5": actual_md5, "sha256": actual_sha256}
    return result


def require_scores(work_dir: Path, model_name: str, benchmarks: dict | None = None) -> dict:
    model_root = work_dir / model_name
    expected = {}
    for dataset in benchmarks or BENCHMARKS:
        pattern = f"*{dataset}*_rating.json" if dataset == "MME-RealWorld-Lite" else f"*{dataset}*_acc.csv"
        matches = list(model_root.glob(pattern))
        if len(matches) != 1:
            raise SystemExit(f"Expected one score artifact for {dataset}, found {matches}")
        expected[dataset] = str(matches[0].resolve())
    return expected


def require_prediction_coverage(
    eval_python: Path, work_dir: Path, model_name: str, data_root: Path, benchmarks: dict | None = None
) -> dict:
    result = {}
    model_root = work_dir / model_name
    checker = """
import hashlib, json, pandas as pd, sys
prediction, source = sys.argv[1:]
pred = pd.read_excel(prediction, usecols=['index'])
src = pd.read_csv(source, sep='\\t', usecols=['index'])
p = [str(x) for x in pred['index'].tolist()]
s = [str(x) for x in src['index'].tolist()]
if len(p) != len(set(p)) or len(s) != len(set(s)) or set(p) != set(s):
    raise SystemExit('prediction/source index coverage mismatch')
canonical = '\\n'.join(sorted(p)) + '\\n'
print(json.dumps({'rows': len(p), 'sorted_index_sha256': hashlib.sha256(canonical.encode()).hexdigest()}))
"""
    for dataset in benchmarks or BENCHMARKS:
        matches = [path for path in model_root.glob(f"*{dataset}.xlsx") if not path.stem.endswith("_score")]
        if len(matches) != 1:
            raise SystemExit(f"Expected one prediction workbook for {dataset}, found {matches}")
        source = data_root / f"{dataset}.tsv"
        checked = subprocess.run(
            [str(eval_python), "-c", checker, str(matches[0]), str(source)],
            capture_output=True,
            text=True,
            check=True,
        )
        coverage = json.loads(checked.stdout)
        result[dataset] = {
            **coverage,
            "path": str(matches[0].resolve()),
            "sha256": sha256_file(matches[0]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True, help="Output label, for example Base or Rotation")
    parser.add_argument("--model-role", choices=("rotation",), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True, help="Exactly one physical GPU index")
    parser.add_argument("--output-root", type=Path, required=True)
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

    lmu_root = output_root / "inputs" / "vlmeval"
    data_manifest_path = lmu_root / "six_benchmark_manifest.json"
    if not data_manifest_path.is_file():
        raise SystemExit(f"Run prepare_evaluation_data.py first: {data_manifest_path}")
    data_manifest = json.loads(data_manifest_path.read_text())
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
        "data": BENCHMARKS,
    }
    model_provenance = model_fingerprint(model_path)
    if model_provenance.get("artifact_type") != "fve_export":
        raise SystemExit("Rotation evaluation requires an exported FVE checkpoint manifest")
    package_freeze = subprocess.run(
        [str(eval_python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    provenance = {
        "config": config,
        "data": dataset_fingerprint(lmu_root, data_manifest),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "evaluator": evaluator_fingerprint(vlmeval_root),
        "model": model_provenance,
        "model_role": args.model_role,
        "package_freeze": package_freeze,
    }
    fingerprint = sha256_bytes(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode())
    attempt_name = f"{args.model_name}-{fingerprint[:16]}"
    work_dir = output_root / "evaluation" / "full" / args.run_id / attempt_name
    config_path = work_dir / "vlmeval_config.json"
    preflight_path = work_dir / "attempt_preflight.json"
    manifest_path = work_dir / "six_benchmark_run_manifest.json"
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
            raise SystemExit(f"Refusing to reuse nonempty evaluation attempt: {work_dir}")
        existing = json.loads(preflight_path.read_text())
        if existing != {"fingerprint": fingerprint, "provenance": provenance}:
            raise SystemExit(f"Evaluation attempt fingerprint mismatch: {work_dir}")
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise SystemExit(f"Evaluation attempt config changed: {config_path}")
        if manifest_path.is_file():
            completed = json.loads(manifest_path.read_text())
            if completed.get("attempt_fingerprint") != fingerprint:
                raise SystemExit(f"Completed evaluation fingerprint mismatch: {manifest_path}")
            if set(completed.get("score_artifacts", {})) != set(BENCHMARKS) or set(
                completed.get("prediction_artifacts", {})
            ) != set(BENCHMARKS):
                raise SystemExit(f"Completed evaluation manifest is incomplete: {manifest_path}")
            for name, artifact in completed.get("score_artifacts", {}).items():
                if sha256_file(Path(artifact)) != completed.get("score_artifact_sha256", {}).get(name):
                    raise SystemExit(f"Completed evaluation score changed: {artifact}")
            for record in completed.get("prediction_artifacts", {}).values():
                if sha256_file(Path(record["path"])) != record.get("sha256"):
                    raise SystemExit(f"Completed evaluation prediction changed: {record['path']}")
            print(json.dumps(completed, indent=2, sort_keys=True))
            return
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        preflight_path.write_text(json.dumps({"fingerprint": fingerprint, "provenance": provenance}, indent=2, sort_keys=True) + "\n")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["LMUData"] = str(lmu_root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_path = work_dir / "six_benchmark.log"
    with log_path.open("a" if args.resume else "x", buffering=1) as log:
        subprocess.run(command, cwd=vlmeval_root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    scores = require_scores(work_dir, args.model_name)
    predictions = require_prediction_coverage(eval_python, work_dir, args.model_name, lmu_root)
    score_hashes = {dataset: sha256_file(Path(path)) for dataset, path in scores.items()}
    manifest = {
        "attempt_fingerprint": fingerprint,
        "benchmarks": list(BENCHMARKS),
        "command": command,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest": str(data_manifest_path),
        "evaluation_role": "full_six_benchmarks",
        "gpu": args.gpu,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "model_role": args.model_role,
        "package_freeze": package_freeze,
        "provenance": provenance,
        "prediction_artifacts": predictions,
        "run_id": args.run_id,
        "schema_version": 2,
        "score_artifact_sha256": score_hashes,
        "score_artifacts": scores,
        "score_input_prediction_sha256": {
            dataset: record["sha256"] for dataset, record in predictions.items()
        },
        "vlmeval_commit": VLMEVAL_COMMIT,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
