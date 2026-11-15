#!/usr/bin/env python3
"""Aggregate compatible full/detail evaluations into paper Tables I--III."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path


TABLE_I = ["MMBench", "SEED-Bench", "V*", "RealWorldQA", "BLINK", "MME", "Average"]
MMBENCH_MAP = {
    "LR": "Logical",
    "RR": "Relation",
    "AR": "Attribute",
    "CP": "Coarse",
    "FP-C": "Cross-Inst.",
    "FP-S": "Single-Inst.",
    "Overall": "Average",
}
SEED_MAP = {
    "Text Understanding": "TU",
    "Visual Reasoning": "VR",
    "Scene Understanding": "SU",
    "Instance Identity": "IId",
    "Instance Interaction": "IIn",
    "Instance Attributes": "IA",
    "Instance Location": "IL",
    "Spatial Relation": "SR",
    "Instances Counting": "IC",
    "Overall": "Average",
}
TABLE_II = ["Logical", "Relation", "Attribute", "Coarse", "Cross-Inst.", "Single-Inst.", "Average"]
TABLE_III = ["TU", "VR", "SU", "IId", "IIn", "IA", "IL", "SR", "IC", "Average"]


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fraction_to_percent(value: str, *, field: str, path: Path) -> Decimal:
    result = Decimal(value)
    if not result.is_finite() or result < 0 or result > 1:
        raise SystemExit(f"Expected a finite fraction in [0,1] for {field} in {path}, got {value!r}")
    return (result * 100).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)


def rounded(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


def read_score_csv(path: Path) -> dict[str, Decimal]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one aggregate row in {path}, found {len(rows)}")
    return {
        key: fraction_to_percent(value, field=key, path=path)
        for key, value in rows[0].items()
        if key != "split" and value not in (None, "")
    }


def read_mme(path: Path) -> Decimal:
    payload = json.loads(path.read_text())
    if "Overall" not in payload:
        raise SystemExit(f"MME-RealWorld-Lite rating lacks Overall: {path}")
    # Pinned VLMEvalKit's get_dimension_rating returns succ_all / sum_all.
    return fraction_to_percent(str(payload["Overall"]), field="Overall", path=path)


def read_manifest(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing run manifest: {path}")
    return path, json.loads(path.read_text())


def score_artifacts(manifest: dict, manifest_path: Path, required: set[str]) -> dict[str, Path]:
    raw = manifest.get("score_artifacts", {})
    hashes = manifest.get("score_artifact_sha256", {})
    if not required.issubset(raw):
        raise SystemExit(f"{manifest_path} lacks score artifacts: {sorted(required - set(raw))}")
    artifacts = {}
    for name in required:
        path = Path(raw[name]).expanduser().resolve()
        if not path.is_file() or hashes.get(name) != sha256_file(path):
            raise SystemExit(f"Score artifact is missing or changed for {name}: {path}")
        artifacts[name] = path
    return artifacts


def generation_signature(manifest: dict) -> dict:
    provenance = manifest.get("provenance", {})
    config = provenance.get("config", {})
    models = list(config.get("model", {}).values())
    if len(models) != 1:
        raise SystemExit("Evaluation provenance must contain exactly one model config")
    generation = dict(models[0])
    generation.pop("model_path", None)
    return {
        "evaluator": provenance.get("evaluator"),
        "generation": generation,
        "package_freeze": manifest.get("package_freeze"),
    }


def validate_run_manifest(path: Path, manifest: dict, required: set[str], schema_version: int) -> None:
    provenance = manifest.get("provenance")
    fingerprint = manifest.get("attempt_fingerprint")
    if manifest.get("schema_version") != schema_version or not isinstance(provenance, dict):
        raise SystemExit(f"Evaluation manifest schema mismatch: {path}")
    expected_fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        raise SystemExit(f"Evaluation attempt fingerprint mismatch: {path}")
    preflight_path = path.parent / "attempt_preflight.json"
    config_path = path.parent / "vlmeval_config.json"
    if (
        not preflight_path.is_file()
        or json.loads(preflight_path.read_text()) != {"fingerprint": fingerprint, "provenance": provenance}
        or not config_path.is_file()
        or json.loads(config_path.read_text()) != provenance.get("config")
    ):
        raise SystemExit(f"Evaluation attempt preflight/config mismatch: {path.parent}")
    if not isinstance(manifest.get("package_freeze"), list) or not manifest["package_freeze"]:
        raise SystemExit(f"Evaluation package provenance is missing: {path}")
    score_artifacts(manifest, path, required)
    for artifact in manifest.get("score_artifacts", {}).values():
        if path.parent not in Path(artifact).expanduser().resolve().parents:
            raise SystemExit(f"Evaluation score is outside its immutable attempt: {artifact}")
    predictions = manifest.get("prediction_artifacts", {})
    if set(predictions) != required:
        raise SystemExit(f"Evaluation prediction set mismatch: {path}")
    for name, record in predictions.items():
        prediction = Path(record.get("path", "")).expanduser().resolve()
        if (
            not prediction.is_file()
            or sha256_file(prediction) != record.get("sha256")
            or int(record.get("rows", 0)) <= 0
            or not record.get("sorted_index_sha256")
        ):
            raise SystemExit(f"Evaluation prediction evidence is missing or changed for {name}: {prediction}")
        if path.parent not in prediction.parents:
            raise SystemExit(f"Evaluation prediction is outside its immutable attempt: {prediction}")
        if manifest.get("score_input_prediction_sha256", {}).get(name) != record.get("sha256"):
            raise SystemExit(f"Evaluation score is not bound to the recorded prediction for {name}")


def validate_full_pair(base: tuple[Path, dict], rotation: tuple[Path, dict]) -> None:
    base_path, base_manifest = base
    rotation_path, rotation_manifest = rotation
    for path, manifest, role in (
        (base_path, base_manifest, "base"),
        (rotation_path, rotation_manifest, "rotation"),
    ):
        if manifest.get("model_role") != role or manifest.get("evaluation_role") != "full_six_benchmarks":
            raise SystemExit(f"Wrong full-evaluation role in {path}; expected {role}")
        validate_run_manifest(
            path,
            manifest,
            {"MMBench_DEV_EN_V11", "SEEDBench_IMG", "VStarBench", "RealWorldQA", "BLINK", "MME-RealWorld-Lite"},
            2,
        )
    if base_manifest.get("run_id") != rotation_manifest.get("run_id"):
        raise SystemExit("Base and Rotation full evaluations have different run IDs")
    if generation_signature(base_manifest) != generation_signature(rotation_manifest):
        raise SystemExit("Base and Rotation use different evaluator or generation protocols")
    if base_manifest.get("provenance", {}).get("config", {}).get("data") != rotation_manifest.get(
        "provenance", {}
    ).get("config", {}).get("data"):
        raise SystemExit("Base and Rotation full evaluations use different benchmark configs")
    base_provenance = base_manifest.get("provenance", {})
    rotation_provenance = rotation_manifest.get("provenance", {})
    for key in ("data", "data_manifest_sha256"):
        if base_provenance.get(key) != rotation_provenance.get(key):
            raise SystemExit(f"Base and Rotation full evaluations use different {key}")


def validate_detail_pair(
    base: tuple[Path, dict],
    rotation: tuple[Path, dict],
    full_base: tuple[Path, dict],
    full_rotation: tuple[Path, dict],
) -> None:
    for (path, manifest), role, (_, full_manifest) in (
        (base, "base", full_base),
        (rotation, "rotation", full_rotation),
    ):
        if manifest.get("model_role") != role or manifest.get("evaluation_role") != "detail_40_percent_heldout":
            raise SystemExit(f"Wrong detail-evaluation role in {path}; expected {role}")
        validate_run_manifest(path, manifest, {"MMBench_DEV_EN_V11", "SEEDBench_IMG"}, 1)
        if manifest.get("run_id") != full_manifest.get("run_id"):
            raise SystemExit(f"Detail/full run ID mismatch for {role}: {path}")
        if generation_signature(manifest) != generation_signature(full_manifest):
            raise SystemExit(f"Detail/full evaluator or generation mismatch for {role}: {path}")
        if manifest.get("provenance", {}).get("model") != full_manifest.get("provenance", {}).get("model"):
            raise SystemExit(f"Detail/full model fingerprint mismatch for {role}: {path}")
    base_manifest = base[1]
    rotation_manifest = rotation[1]
    for key in ("split_id", "split_manifest_sha256"):
        if base_manifest.get(key) != rotation_manifest.get(key):
            raise SystemExit(f"Base and Rotation detail evaluations use different {key}")
    if base_manifest.get("provenance", {}).get("split") != rotation_manifest.get("provenance", {}).get("split"):
        raise SystemExit("Base and Rotation detail evaluations use different exact held-out data")
    if base_manifest.get("provenance", {}).get("config", {}).get("data") != rotation_manifest.get(
        "provenance", {}
    ).get("config", {}).get("data"):
        raise SystemExit("Base and Rotation detail evaluations use different benchmark configs")


def load_run(
    full: tuple[Path, dict], detail: tuple[Path, dict] | None = None
) -> dict:
    manifest_path, manifest = full
    full_required = {"MMBench_DEV_EN_V11", "SEEDBench_IMG", "VStarBench", "RealWorldQA", "BLINK", "MME-RealWorld-Lite"}
    artifacts = score_artifacts(manifest, manifest_path, full_required)
    mmbench = read_score_csv(artifacts["MMBench_DEV_EN_V11"])
    seed = read_score_csv(artifacts["SEEDBench_IMG"])
    detail_artifacts = artifacts
    detail_path = None
    if detail is not None:
        detail_path, detail_manifest = detail
        detail_artifacts = score_artifacts(
            detail_manifest, detail_path, {"MMBench_DEV_EN_V11", "SEEDBench_IMG"}
        )
    mmbench_detail = read_score_csv(detail_artifacts["MMBench_DEV_EN_V11"])
    seed_detail = read_score_csv(detail_artifacts["SEEDBench_IMG"])
    for source in MMBENCH_MAP:
        if source not in mmbench_detail:
            raise SystemExit(f"MMBench detail artifact lacks {source}")
    for source in SEED_MAP:
        if source not in seed_detail:
            raise SystemExit(f"SEED detail artifact lacks {source}")
    benchmark_scores = {
        "MMBench": mmbench["Overall"],
        "SEED-Bench": seed["Overall"],
        "V*": read_score_csv(artifacts["VStarBench"])["Overall"],
        "RealWorldQA": read_score_csv(artifacts["RealWorldQA"])["Overall"],
        "BLINK": read_score_csv(artifacts["BLINK"])["Overall"],
        "MME": read_mme(artifacts["MME-RealWorld-Lite"]),
    }
    benchmark_scores["Average"] = sum(benchmark_scores.values()) / Decimal(len(benchmark_scores))
    return {
        "detail_manifest": str(detail_path) if detail_path else None,
        "manifest": str(manifest_path),
        "model_name": manifest["model_name"],
        "table_i": benchmark_scores,
        "table_ii": {table: mmbench_detail[source] for source, table in MMBENCH_MAP.items()},
        "table_iii": {table: seed_detail[source] for source, table in SEED_MAP.items()},
    }


def decimal_strings(values: dict[str, Decimal], columns: list[str]) -> dict[str, str]:
    return {column: format(rounded(values[column]), ".2f") for column in columns}


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Category", "Model", *columns])
        writer.writeheader()
        writer.writerows(rows)


def latex_table(caption: str, columns: list[str], rows: list[dict]) -> str:
    header = " & ".join(["Category", "Model", *columns]) + r" \\"
    body = [" & ".join([row["Category"], row["Model"], *[row[column] for column in columns]]) + r" \\" for row in rows]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\begin{tabular}{ll" + "c" * len(columns) + "}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--base-detail-manifest", type=Path, default=None)
    parser.add_argument("--rotation-detail-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-display-name", default="Qwen2.5-VL-3B")
    parser.add_argument("--rotation-display-name", default="Rotation")
    args = parser.parse_args()

    if (args.base_detail_manifest is None) != (args.rotation_detail_manifest is None):
        raise SystemExit("Supply both detail manifests or neither; unpaired detail results are forbidden")
    full_base = read_manifest(args.base_manifest)
    full_rotation = read_manifest(args.rotation_manifest)
    validate_full_pair(full_base, full_rotation)
    detail_base = read_manifest(args.base_detail_manifest) if args.base_detail_manifest else None
    detail_rotation = read_manifest(args.rotation_detail_manifest) if args.rotation_detail_manifest else None
    if detail_base is not None and detail_rotation is not None:
        validate_detail_pair(detail_base, detail_rotation, full_base, full_rotation)

    base = load_run(full_base, detail_base)
    rotation = load_run(full_rotation, detail_rotation)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "base_detail": None if detail_base is None else {"path": str(detail_base[0]), "sha256": sha256_file(detail_base[0])},
        "base_full": {"path": str(full_base[0]), "sha256": sha256_file(full_base[0])},
        "rotation_detail": None if detail_rotation is None else {"path": str(detail_rotation[0]), "sha256": sha256_file(detail_rotation[0])},
        "rotation_full": {"path": str(full_rotation[0]), "sha256": sha256_file(full_rotation[0])},
    }
    detail_scope = "full set" if detail_base is None else "40\\% held-out set"
    table_specs = [
        ("table_i", TABLE_I, "Test performance (\\%) on VQA benchmarks."),
        ("table_ii", TABLE_II, f"Performance (\\%) on the MMBench {detail_scope}."),
        ("table_iii", TABLE_III, f"Performance (\\%) on the SEED-Bench {detail_scope}."),
    ]
    bundle = {
        "detail_protocol": "full benchmark" if detail_base is None else "deterministic 40% held-out split",
        "inputs": inputs,
        "run_id": full_base[1]["run_id"],
        "schema_version": 2,
        "tables": {},
    }
    with tempfile.TemporaryDirectory(prefix=".paper-tables-", dir=output_dir) as temporary:
        staging = Path(temporary)
        latex_parts = []
        for table_key, columns, caption in table_specs:
            rows = [
                {"Category": "Base", "Model": args.base_display_name, **decimal_strings(base[table_key], columns)},
                {"Category": "SSL4RL-3B", "Model": args.rotation_display_name, **decimal_strings(rotation[table_key], columns)},
            ]
            improvement = {
                column: format(rounded(rotation[table_key][column] - base[table_key][column]), ".2f")
                for column in columns
            }
            table_path = staging / f"{table_key}.csv"
            write_csv(table_path, columns, rows)
            latex_parts.append(latex_table(caption, columns, rows))
            bundle["tables"][table_key] = {
                "columns": columns,
                "rows": rows,
                "rotation_minus_base": improvement,
            }
        tex_path = staging / "paper_tables.tex"
        tex_path.write_text("\n".join(latex_parts))
        bundle["artifact_sha256"] = {
            path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()
        }
        manifest_path = staging / "paper_tables.json"
        manifest_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        for path in sorted(staging.iterdir()):
            os.replace(path, output_dir / path.name)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
