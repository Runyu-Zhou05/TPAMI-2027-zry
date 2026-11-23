#!/usr/bin/env python3
"""Run immutable chain-of-thought MathVista and MMMU evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from deepseek_credential import resolve_deepseek_api_key
from prepare_reasoning_data import DATASETS, digest
from run_six_benchmarks import (
    VLMEVAL_COMMIT,
    evaluator_fingerprint,
    model_fingerprint,
    require_executable,
    require_prediction_coverage,
    sha256_bytes,
    sha256_file,
)


BENCHMARKS = {
    "MathVista_MINI": {"class": "MathVista", "dataset": "MathVista_MINI"},
    "MMMU_DEV_VAL": {"class": "MMMUDataset", "dataset": "MMMU_DEV_VAL"},
}
JUDGES = ("deepseek-chat", "gpt-4-0125", "gpt-4o", "gpt-4o-0806", "gpt-4o-mini")
MODEL_ROLES = ("position", "rotation", "jigsaw", "contrastive")


def data_fingerprint(lmu_root: Path, manifest: dict) -> dict:
    result = {}
    for name in BENCHMARKS:
        info = manifest.get("benchmarks", {}).get(name, {})
        canonical = (lmu_root / f"{name}.tsv").resolve()
        recorded = Path(info.get("path", "")).expanduser().resolve()
        if canonical != recorded or not canonical.is_file():
            raise SystemExit(f"Non-canonical reasoning dataset path for {name}: {recorded}")
        actual_md5 = digest(canonical, "md5")
        actual_sha256 = digest(canonical, "sha256")
        if (
            actual_md5 != DATASETS[name]["md5"]
            or actual_md5 != info.get("md5")
            or actual_sha256 != info.get("sha256")
        ):
            raise SystemExit(f"Reasoning dataset checksum mismatch for {name}: {canonical}")
        result[name] = {
            "bytes": canonical.stat().st_size,
            "md5": actual_md5,
            "sha256": actual_sha256,
        }
    return result


def one_match(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected one {label}, found {matches}")
    return matches[0].resolve()


def reasoning_coverage(eval_python: Path, predictions: dict[str, dict]) -> dict:
    checker = r'''
import json, pandas as pd, sys
result = {}
for item in sys.argv[1:]:
    name, path = item.split("=", 1)
    data = pd.read_excel(path, usecols=["prediction"])
    values = [str(x) for x in data["prediction"].tolist()]
    nonempty = sum(bool(x.strip()) and x.lower() != "nan" for x in values)
    formatted = sum(
        "<think>" in x and "</think>" in x and "<answer>" in x and "</answer>" in x
        for x in values
    )
    result[name] = {
        "formatted_reasoning_rows": formatted,
        "formatted_reasoning_rate": formatted / len(values) if values else 0.0,
        "nonempty_rows": nonempty,
        "rows": len(values),
    }
print(json.dumps(result))
'''
    arguments = [f"{name}={record['path']}" for name, record in predictions.items()]
    completed = subprocess.run(
        [str(eval_python), "-c", checker, *arguments], capture_output=True, text=True, check=True
    )
    result = json.loads(completed.stdout)
    for name, record in result.items():
        if int(record["rows"]) <= 0 or int(record["nonempty_rows"]) != int(record["rows"]):
            raise SystemExit(f"Reasoning predictions are empty or incomplete for {name}: {record}")
    return result


def score_summary(eval_python: Path, mathvista_score: Path, mmmu_score: Path) -> dict:
    checker = r'''
import json, math, pandas as pd, sys
mathvista = pd.read_csv(sys.argv[1])
mmmu = pd.read_csv(sys.argv[2])
overall = mathvista[mathvista["Task&Skill"] == "Overall"]
if len(overall) != 1 or "acc" not in overall:
    raise SystemExit("MathVista score lacks one Overall accuracy row")
mathvista_acc = float(overall.iloc[0]["acc"])
if len(mmmu) != 1 or "Overall" not in mmmu:
    raise SystemExit("MMMU score lacks one Overall row")
mmmu_acc = float(mmmu.iloc[0]["Overall"])
if not math.isfinite(mathvista_acc) or not 0 <= mathvista_acc <= 100:
    raise SystemExit("MathVista Overall is outside [0,100]")
if not math.isfinite(mmmu_acc) or not 0 <= mmmu_acc <= 1:
    raise SystemExit("MMMU Overall is outside [0,1]")
print(json.dumps({"MathVista_MINI": mathvista_acc, "MMMU_DEV_VAL": mmmu_acc * 100.0}))
'''
    completed = subprocess.run(
        [str(eval_python), "-c", checker, str(mathvista_score), str(mmmu_score)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def validate_completed(manifest_path: Path, fingerprint: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("attempt_fingerprint") != fingerprint or manifest.get("schema_version") != 1:
        raise SystemExit(f"Completed reasoning attempt identity mismatch: {manifest_path}")
    for section in ("prediction_artifacts", "score_artifacts"):
        records = manifest.get(section, {})
        if set(records) != set(BENCHMARKS):
            raise SystemExit(f"Completed reasoning attempt lacks {section}: {manifest_path}")
        for name, record in records.items():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise SystemExit(f"Completed reasoning artifact changed for {name}: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-role", choices=MODEL_ROLES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mathvista-judge", choices=JUDGES, default="deepseek-chat")
    parser.add_argument(
        "--judge-api-key-file",
        type=Path,
        default=Path(os.environ["DEEPSEEK_API_KEY_FILE"]) if os.environ.get("DEEPSEEK_API_KEY_FILE") else None,
    )
    parser.add_argument("--deepseek-api-base", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    for label, value in (("model-name", args.model_name), ("run-id", args.run_id)):
        if not value or any(character not in allowed for character in value):
            raise SystemExit(f"--{label} contains unsupported characters: {value!r}")
    if "," in args.gpu or not args.gpu.isdigit():
        raise SystemExit("--gpu must be exactly one physical GPU index")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    output_root = args.output_root.expanduser().resolve()
    eval_root = (args.eval_root or output_root / "tools" / "vlmevalkit").expanduser().resolve()
    eval_python = require_executable(eval_root / "env" / "bin" / "python")
    vlmeval_root = eval_root / f"VLMEvalKit-{VLMEVAL_COMMIT}"
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Invalid model path: {model_path}")
    lmu_root = output_root / "inputs" / "vlmeval"
    data_manifest_path = lmu_root / "reasoning_benchmark_manifest.json"
    if not data_manifest_path.is_file():
        raise SystemExit(f"Run prepare_reasoning_data.py first: {data_manifest_path}")
    data_manifest = json.loads(data_manifest_path.read_text())

    model_config = {
        "class": "Qwen2VLChat",
        "model_path": str(model_path),
        "min_pixels": 1280 * 28 * 28,
        "max_pixels": 16384 * 28 * 28,
        "max_new_tokens": args.max_new_tokens,
        "top_p": 0.001,
        "top_k": 1,
        "temperature": 0.01,
        "repetition_penalty": 1.0,
        "use_custom_prompt": False,
    }
    configs = {
        "MathVista_MINI": {
            "model": {args.model_name: model_config},
            "data": {"MathVista_MINI": BENCHMARKS["MathVista_MINI"]},
        },
        "MMMU_DEV_VAL": {
            "model": {args.model_name: model_config},
            "data": {"MMMU_DEV_VAL": BENCHMARKS["MMMU_DEV_VAL"]},
        },
    }
    model_provenance = model_fingerprint(model_path)
    artifact_type = model_provenance.get("artifact_type")
    exported_role = model_provenance.get("model_role")
    if artifact_type == "fve_export":
        if args.model_role != "rotation" or exported_role != "rotation":
            raise SystemExit("FVE reasoning evaluation requires the Rotation role")
    elif artifact_type == "ssl4rl_export":
        if exported_role != args.model_role:
            raise SystemExit(
                f"SSL4RL export role {exported_role!r} does not match --model-role {args.model_role!r}"
            )
    else:
        raise SystemExit("Reasoning evaluation requires a validated FVE or SSL4RL export manifest")
    packages = subprocess.run(
        [str(eval_python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    provenance = {
        "configs": configs,
        "data": data_fingerprint(lmu_root, data_manifest),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "evaluator": evaluator_fingerprint(vlmeval_root),
        "judges": {"MathVista_MINI": args.mathvista_judge, "MMMU_DEV_VAL": "exact_matching"},
        "judge_api": {
            "MathVista_MINI": args.deepseek_api_base if args.mathvista_judge == "deepseek-chat" else "VLMEvalKit default"
        },
        "model": model_provenance,
        "model_role": args.model_role,
        "package_freeze": packages,
        "reasoning_format": "<think>complete reasoning</think> followed by <answer>final answer</answer>",
    }
    fingerprint = sha256_bytes(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode())
    work_dir = output_root / "evaluation" / "reasoning" / args.run_id / f"{args.model_name}-{fingerprint[:16]}"
    preflight_path = work_dir / "attempt_preflight.json"
    manifest_path = work_dir / "reasoning_benchmark_run_manifest.json"
    config_paths = {name: work_dir / f"vlmeval_config_{name}.json" for name in BENCHMARKS}
    commands = {
        "MMMU_DEV_VAL": [
            str(eval_python),
            str(vlmeval_root / "run.py"),
            "--config",
            str(config_paths["MMMU_DEV_VAL"]),
            "--work-dir",
            str(work_dir),
            "--judge",
            "exact_matching",
        ],
        "MathVista_MINI": [
            str(eval_python),
            str(vlmeval_root / "run.py"),
            "--config",
            str(config_paths["MathVista_MINI"]),
            "--work-dir",
            str(work_dir),
            "--judge",
            args.mathvista_judge,
        ],
    }
    if args.resume:
        for command in commands.values():
            command.append("--reuse")
    if args.dry_run:
        print(json.dumps({"attempt": str(work_dir), "commands": commands, "fingerprint": fingerprint}, indent=2))
        return
    judge_key = None
    judge_credential_source = None
    if args.mathvista_judge == "deepseek-chat":
        judge_key, judge_credential_source = resolve_deepseek_api_key(args.judge_api_key_file)
    expected_preflight = {"fingerprint": fingerprint, "provenance": provenance}
    if work_dir.exists() and any(work_dir.iterdir()):
        if not args.resume or not preflight_path.is_file():
            raise SystemExit(f"Refusing to reuse nonempty reasoning attempt: {work_dir}")
        if json.loads(preflight_path.read_text()) != expected_preflight:
            raise SystemExit(f"Reasoning attempt fingerprint mismatch: {work_dir}")
        for name, config_path in config_paths.items():
            if not config_path.is_file() or json.loads(config_path.read_text()) != configs[name]:
                raise SystemExit(f"Reasoning attempt config changed: {config_path}")
        if manifest_path.is_file():
            print(json.dumps(validate_completed(manifest_path, fingerprint), indent=2, sort_keys=True))
            return
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(expected_preflight, indent=2, sort_keys=True) + "\n")
        for name, config_path in config_paths.items():
            config_path.write_text(json.dumps(configs[name], indent=2, sort_keys=True) + "\n")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["LMUData"] = str(lmu_root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if args.mathvista_judge == "deepseek-chat":
        environment["OPENAI_API_KEY"] = judge_key
        environment["OPENAI_API_BASE"] = args.deepseek_api_base
    for name in ("MMMU_DEV_VAL", "MathVista_MINI"):
        with (work_dir / f"{name}.log").open("a" if args.resume else "x", buffering=1) as log:
            subprocess.run(
                commands[name], cwd=vlmeval_root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    model_root = work_dir / args.model_name
    score_paths = {
        "MathVista_MINI": one_match(model_root, "*MathVista_MINI*_score.csv", "MathVista score file"),
        "MMMU_DEV_VAL": one_match(model_root, "*MMMU_DEV_VAL*_acc.csv", "MMMU score file"),
    }
    predictions = require_prediction_coverage(eval_python, work_dir, args.model_name, lmu_root, BENCHMARKS)
    scores = {
        name: {"path": str(path), "sha256": sha256_file(path)} for name, path in score_paths.items()
    }
    manifest = {
        "attempt_fingerprint": fingerprint,
        "benchmarks": list(BENCHMARKS),
        "commands": commands,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "reasoning_two_benchmarks",
        "gpu": args.gpu,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "model_role": args.model_role,
        "judge_credential_source": judge_credential_source,
        "package_freeze": packages,
        "prediction_artifacts": predictions,
        "provenance": provenance,
        "reasoning_coverage": reasoning_coverage(eval_python, predictions),
        "run_id": args.run_id,
        "schema_version": 1,
        "score_artifacts": scores,
        "score_input_prediction_sha256": {
            name: record["sha256"] for name, record in predictions.items()
        },
        "scores_percent": score_summary(
            eval_python, score_paths["MathVista_MINI"], score_paths["MMMU_DEV_VAL"]
        ),
        "vlmeval_commit": VLMEVAL_COMMIT,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
