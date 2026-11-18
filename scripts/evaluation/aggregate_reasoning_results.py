#!/usr/bin/env python3
"""Publish Rotation-only MathVista/MMMU reasoning results."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rotation_path, rotation = load_run(args.rotation_manifest, "rotation")

    rows = []
    summary = {}
    for name in BENCHMARKS:
        rotation_score = float(rotation["scores_percent"][name])
        row = {
            "Benchmark": name,
            "Rotation": f"{rotation_score:.2f}",
        }
        rows.append(row)
        summary[name] = {
            "rotation": rotation_score,
            "rotation_reasoning_coverage": rotation["reasoning_coverage"][name],
        }
    bundle = {
        "benchmarks": summary,
        "rotation_manifest": {"path": str(rotation_path), "sha256": sha256_file(rotation_path)},
        "run_id": rotation["run_id"],
        "schema_version": 1,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reasoning-results-", dir=output_dir) as temporary:
        staging = Path(temporary)
        csv_path = staging / "reasoning_results.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Benchmark", "Rotation"])
            writer.writeheader()
            writer.writerows(rows)
        (staging / "reasoning_results.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        for path in staging.iterdir():
            os.replace(path, output_dir / path.name)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
