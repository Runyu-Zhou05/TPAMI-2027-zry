#!/usr/bin/env python3
"""Aggregate two six-benchmark runs into SSL4RL paper Tables I--III."""

from __future__ import annotations

import argparse
import csv
import json
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


def percent(value) -> Decimal:
    result = Decimal(str(value))
    if abs(result) <= Decimal("1.0000001"):
        result *= 100
    # Remove binary-float serialization noise from CSV/JSON scores before
    # computing cross-benchmark means.
    return result.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)


def rounded(value: Decimal) -> Decimal:
    # Match the paper table convention at exact .005 ties (for example,
    # Table I Base average 53.365 -> 53.36).
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


def read_score_csv(path: Path) -> dict[str, Decimal]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one aggregate row in {path}, found {len(rows)}")
    return {key: percent(value) for key, value in rows[0].items() if key != "split" and value not in (None, "")}


def read_mme(path: Path) -> Decimal:
    payload = json.loads(path.read_text())
    if "Overall" not in payload:
        raise SystemExit(f"MME-RealWorld-Lite rating lacks Overall: {path}")
    return percent(payload["Overall"])


def load_run(manifest_path: Path, mmbench_detail: Path | None = None, seed_detail: Path | None = None) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    artifacts = {key: Path(value) for key, value in manifest["score_artifacts"].items()}
    for key, path in artifacts.items():
        if not path.is_file():
            raise SystemExit(f"Missing score artifact for {key}: {path}")

    mmbench = read_score_csv(artifacts["MMBench_DEV_EN_V11"])
    seed = read_score_csv(artifacts["SEEDBench_IMG"])
    mmbench_detail_scores = read_score_csv(mmbench_detail.expanduser().resolve()) if mmbench_detail else mmbench
    seed_detail_scores = read_score_csv(seed_detail.expanduser().resolve()) if seed_detail else seed
    benchmark_scores = {
        "MMBench": mmbench["Overall"],
        "SEED-Bench": seed["Overall"],
        "V*": read_score_csv(artifacts["VStarBench"])["Overall"],
        "RealWorldQA": read_score_csv(artifacts["RealWorldQA"])["Overall"],
        "BLINK": read_score_csv(artifacts["BLINK"])["Overall"],
        "MME": read_mme(artifacts["MME-RealWorld-Lite"]),
    }
    benchmark_scores["Average"] = sum(benchmark_scores.values()) / Decimal(len(benchmark_scores))
    mmbench_table = {
        table_name: mmbench_detail_scores[source_name] for source_name, table_name in MMBENCH_MAP.items()
    }
    seed_table = {table_name: seed_detail_scores[source_name] for source_name, table_name in SEED_MAP.items()}
    return {
        "manifest": str(manifest_path),
        "model_name": manifest["model_name"],
        "table_i": benchmark_scores,
        "table_ii": mmbench_table,
        "table_iii": seed_table,
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
    parser.add_argument("--base-mmbench-detail", type=Path, default=None)
    parser.add_argument("--rotation-mmbench-detail", type=Path, default=None)
    parser.add_argument("--base-seed-detail", type=Path, default=None)
    parser.add_argument("--rotation-seed-detail", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-display-name", default="Qwen2.5-VL-3B")
    parser.add_argument("--rotation-display-name", default="Rotation")
    args = parser.parse_args()

    base = load_run(args.base_manifest, args.base_mmbench_detail, args.base_seed_detail)
    rotation = load_run(
        args.rotation_manifest, args.rotation_mmbench_detail, args.rotation_seed_detail
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table_specs = [
        ("table_i", TABLE_I, "Test performance (\\%) on VQA benchmarks."),
        ("table_ii", TABLE_II, "Performance (\\%) on the MMBench DEV-EN set."),
        ("table_iii", TABLE_III, "Performance (\\%) on the SEED-Bench image set."),
    ]
    bundle = {
        "base_manifest": base["manifest"],
        "rotation_manifest": rotation["manifest"],
        "detail_score_overrides": {
            "base_mmbench": str(args.base_mmbench_detail.resolve()) if args.base_mmbench_detail else None,
            "rotation_mmbench": str(args.rotation_mmbench_detail.resolve()) if args.rotation_mmbench_detail else None,
            "base_seed": str(args.base_seed_detail.resolve()) if args.base_seed_detail else None,
            "rotation_seed": str(args.rotation_seed_detail.resolve()) if args.rotation_seed_detail else None,
        },
        "tables": {},
    }
    latex_parts = []
    for table_key, columns, caption in table_specs:
        rows = [
            {
                "Category": "Base",
                "Model": args.base_display_name,
                **decimal_strings(base[table_key], columns),
            },
            {
                "Category": "SSL4RL-3B",
                "Model": args.rotation_display_name,
                **decimal_strings(rotation[table_key], columns),
            },
        ]
        improvement = {
            column: format(rounded(rotation[table_key][column] - base[table_key][column]), ".2f")
            for column in columns
        }
        write_csv(output_dir / f"{table_key}.csv", columns, rows)
        latex_parts.append(latex_table(caption, columns, rows))
        bundle["tables"][table_key] = {"columns": columns, "rows": rows, "rotation_minus_base": improvement}

    (output_dir / "paper_tables.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    (output_dir / "paper_tables.tex").write_text("\n".join(latex_parts))
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
