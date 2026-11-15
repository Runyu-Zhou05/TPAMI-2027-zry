#!/usr/bin/env python3
"""Run the six paper VQA benchmarks with pinned VLMEvalKit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


VLMEVAL_COMMIT = "0a11917eede83742389b03d7f11fc252aabdf6ef"
BENCHMARKS = {
    "MMBench_DEV_EN_V11": {"class": "ImageMCQDataset", "dataset": "MMBench_DEV_EN_V11"},
    "SEEDBench_IMG": {"class": "ImageMCQDataset", "dataset": "SEEDBench_IMG"},
    "VStarBench": {"class": "ImageMCQDataset", "dataset": "VStarBench"},
    "RealWorldQA": {"class": "ImageMCQDataset", "dataset": "RealWorldQA"},
    "BLINK": {"class": "ImageMCQDataset", "dataset": "BLINK"},
    "MME-RealWorld-Lite": {"class": "MMERealWorld", "dataset": "MME-RealWorld-Lite"},
}


def require_executable(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Executable is unavailable: {path}")
    return path


def require_scores(work_dir: Path, model_name: str) -> dict:
    model_root = work_dir / model_name
    expected = {}
    for dataset in BENCHMARKS:
        if dataset == "MME-RealWorld-Lite":
            matches = list(model_root.glob(f"*{dataset}*_rating.json"))
        else:
            matches = list(model_root.glob(f"*{dataset}*_acc.csv"))
        if len(matches) != 1:
            raise SystemExit(f"Expected one score artifact for {dataset}, found {matches}")
        expected[dataset] = str(matches[0].resolve())
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True, help="Output label, for example Base or Rotation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True, help="Exactly one physical GPU index")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=None)
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
    marker = vlmeval_root / ".pinned-commit"
    if not marker.is_file() or marker.read_text().strip() != VLMEVAL_COMMIT:
        raise SystemExit(f"Pinned VLMEvalKit is unavailable under {eval_root}")
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Invalid model path: {model_path}")

    lmu_root = output_root / "inputs" / "vlmeval"
    data_manifest_path = lmu_root / "six_benchmark_manifest.json"
    if not data_manifest_path.is_file():
        raise SystemExit(f"Run prepare_evaluation_data.py first: {data_manifest_path}")
    data_manifest = json.loads(data_manifest_path.read_text())
    missing = sorted(set(BENCHMARKS) - set(data_manifest.get("benchmarks", {})))
    if missing:
        raise SystemExit(f"Evaluation data manifest is missing: {missing}")
    for dataset in BENCHMARKS:
        dataset_path = Path(data_manifest["benchmarks"][dataset]["path"])
        if not dataset_path.is_file():
            raise SystemExit(f"Evaluation dataset file is missing for {dataset}: {dataset_path}")

    work_dir = output_root / "evaluation" / "full" / args.run_id / args.model_name
    work_dir.mkdir(parents=True, exist_ok=True)
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
    config_path = work_dir / "vlmeval_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    command = [
        str(eval_python),
        str(vlmeval_root / "run.py"),
        "--config",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--judge",
        "exact_matching",
        "--reuse",
    ]
    if args.dry_run:
        print(" ".join(command))
        return

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["LMUData"] = str(lmu_root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_path = work_dir / "six_benchmark.log"
    with log_path.open("w", buffering=1) as log:
        subprocess.run(command, cwd=vlmeval_root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    scores = require_scores(work_dir, args.model_name)
    manifest = {
        "benchmarks": list(BENCHMARKS),
        "command": command,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest": str(data_manifest_path),
        "gpu": args.gpu,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "run_id": args.run_id,
        "score_artifacts": scores,
        "vlmeval_commit": VLMEVAL_COMMIT,
    }
    manifest_path = work_dir / "six_benchmark_run_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
