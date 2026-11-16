#!/usr/bin/env python3
"""Compare compatible Base and Rotation MathVista/MMMU reasoning runs."""

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


def protocol_signature(manifest: dict) -> dict:
    provenance = manifest["provenance"]
    configs = {}
    for name, original in provenance["configs"].items():
        config = json.loads(json.dumps(original))
        model = next(iter(config.pop("model").values()))
        model.pop("model_path", None)
        configs[name] = {**config, "model": model}
    return {
        "configs": configs,
        "data": provenance["data"],
        "data_manifest_sha256": provenance["data_manifest_sha256"],
        "evaluator": provenance["evaluator"],
        "judges": provenance["judges"],
        "package_freeze": provenance["package_freeze"],
        "reasoning_format": provenance["reasoning_format"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_path, base = load_run(args.base_manifest, "base")
    rotation_path, rotation = load_run(args.rotation_manifest, "rotation")
    if base.get("run_id") != rotation.get("run_id"):
        raise SystemExit("Base and Rotation reasoning runs have different run IDs")
    if protocol_signature(base) != protocol_signature(rotation):
        raise SystemExit("Base and Rotation reasoning runs use different protocols")

    rows = []
    summary = {}
    for name in BENCHMARKS:
        base_score = float(base["scores_percent"][name])
        rotation_score = float(rotation["scores_percent"][name])
        row = {
            "Benchmark": name,
            "Base": f"{base_score:.2f}",
            "Rotation": f"{rotation_score:.2f}",
            "Rotation-Base": f"{rotation_score - base_score:.2f}",
        }
        rows.append(row)
        summary[name] = {
            "base": base_score,
            "base_reasoning_coverage": base["reasoning_coverage"][name],
            "rotation": rotation_score,
            "rotation_minus_base": rotation_score - base_score,
            "rotation_reasoning_coverage": rotation["reasoning_coverage"][name],
        }
    bundle = {
        "base_manifest": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "benchmarks": summary,
        "rotation_manifest": {"path": str(rotation_path), "sha256": sha256_file(rotation_path)},
        "run_id": base["run_id"],
        "schema_version": 1,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reasoning-results-", dir=output_dir) as temporary:
        staging = Path(temporary)
        csv_path = staging / "reasoning_results.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Benchmark", "Base", "Rotation", "Rotation-Base"])
            writer.writeheader()
            writer.writerows(rows)
        (staging / "reasoning_results.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        for path in staging.iterdir():
            os.replace(path, output_dir / path.name)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
