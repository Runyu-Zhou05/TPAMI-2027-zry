#!/usr/bin/env python3
"""Publish four-task SSL4RL MathVista/MMMU reasoning results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


BENCHMARKS = ("MathVista_MINI", "MMMU_DEV_VAL")
MODEL_ROLES = ("position", "rotation", "jigsaw", "contrastive")
DISPLAY_NAMES = {
    "position": "Position",
    "rotation": "Rotation",
    "jigsaw": "Jigsaw",
    "contrastive": "Contrastive",
}


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_run(path: Path, role: str) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing {role} reasoning manifest: {path}")
    manifest = json.loads(path.read_text())
    provenance = manifest.get("provenance", {})
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        manifest.get("schema_version") != 1
        or manifest.get("evaluation_role") != "reasoning_two_benchmarks"
        or manifest.get("model_role") != role
        or manifest.get("attempt_fingerprint") != fingerprint
        or tuple(manifest.get("benchmarks", [])) != BENCHMARKS
    ):
        raise SystemExit(f"Invalid {role} reasoning manifest identity: {path}")
    preflight = path.parent / "attempt_preflight.json"
    if not preflight.is_file() or json.loads(preflight.read_text()) != {
        "fingerprint": fingerprint,
        "provenance": provenance,
    }:
        raise SystemExit(f"Reasoning preflight mismatch: {path.parent}")
    for name, config in provenance.get("configs", {}).items():
        config_path = path.parent / f"vlmeval_config_{name}.json"
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise SystemExit(f"Reasoning config mismatch: {config_path}")
    for section in ("prediction_artifacts", "score_artifacts"):
        records = manifest.get(section, {})
        if set(records) != set(BENCHMARKS):
            raise SystemExit(f"Reasoning manifest lacks {section}: {path}")
        for name, record in records.items():
            artifact = Path(record.get("path", "")).expanduser().resolve()
            if (
                not artifact.is_file()
                or path.parent not in artifact.parents
                or sha256_file(artifact) != record.get("sha256")
            ):
                raise SystemExit(f"Reasoning artifact is missing, changed, or outside its attempt: {artifact}")
    for name in BENCHMARKS:
        if (
            manifest.get("score_input_prediction_sha256", {}).get(name)
            != manifest["prediction_artifacts"][name]["sha256"]
        ):
            raise SystemExit(f"Reasoning score is not bound to its prediction for {name}")
        score = float(manifest.get("scores_percent", {}).get(name, float("nan")))
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise SystemExit(f"Reasoning score is outside [0,100] for {name}: {score}")
    return path, manifest


def evaluation_signature(manifest: dict) -> dict:
    provenance = manifest["provenance"]
    generation = {}
    for benchmark, config in provenance["configs"].items():
        model_records = list(config["model"].values())
        if len(model_records) != 1:
            raise SystemExit(f"Reasoning config must contain exactly one model: {benchmark}")
        model_config = dict(model_records[0])
        model_config.pop("model_path", None)
        generation[benchmark] = {"model": model_config, "data": config["data"]}
    return {
        "data": provenance["data"],
        "data_manifest_sha256": provenance["data_manifest_sha256"],
        "evaluator": provenance["evaluator"],
        "generation": generation,
        "judge_api": provenance["judge_api"],
        "judges": provenance["judges"],
        "package_freeze": provenance["package_freeze"],
        "reasoning_format": provenance["reasoning_format"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in MODEL_ROLES:
        parser.add_argument(f"--{role}-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected_roles = tuple(role for role in MODEL_ROLES if getattr(args, f"{role}_manifest") is not None)
    if set(selected_roles) not in ({"rotation"}, set(MODEL_ROLES)):
        raise SystemExit("Provide either only --rotation-manifest or all four SSL4RL task manifests")
    runs = {
        role: load_run(getattr(args, f"{role}_manifest"), role)
        for role in selected_roles
    }
    signatures = {role: evaluation_signature(manifest) for role, (_, manifest) in runs.items()}
    reference = signatures[selected_roles[0]]
    for role in selected_roles[1:]:
        if signatures[role] != reference:
            raise SystemExit(f"Reasoning evaluation protocol mismatch for {role}")

    rows = []
    summary = {}
    for name in BENCHMARKS:
        row = {"Benchmark": name}
        benchmark_summary = {}
        for role in selected_roles:
            _, manifest = runs[role]
            score = float(manifest["scores_percent"][name])
            row[DISPLAY_NAMES[role]] = f"{score:.2f}"
            benchmark_summary[role] = {
                "score": score,
                "reasoning_coverage": manifest["reasoning_coverage"][name],
            }
        rows.append(row)
        summary[name] = benchmark_summary
    bundle = {
        "benchmarks": summary,
        "manifests": {
            role: {"path": str(path), "sha256": sha256_file(path), "run_id": manifest["run_id"]}
            for role, (path, manifest) in runs.items()
        },
        "schema_version": 2,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reasoning-results-", dir=output_dir) as temporary:
        staging = Path(temporary)
        csv_path = staging / "reasoning_results.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["Benchmark", *[DISPLAY_NAMES[role] for role in selected_roles]],
            )
            writer.writeheader()
            writer.writerows(rows)
        (staging / "reasoning_results.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        for path in staging.iterdir():
            os.replace(path, output_dir / path.name)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
