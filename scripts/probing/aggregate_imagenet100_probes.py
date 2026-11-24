#!/usr/bin/env python3
"""Aggregate compatible Base and four-task ImageNet-100 linear probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROLES = ("base", "rotation", "position", "jigsaw", "contrastive")
DISPLAY = {
    "base": "Base",
    "rotation": "Rotation",
    "position": "Position",
    "jigsaw": "Jigsaw",
    "contrastive": "Contrastive",
}
LEVELS = ("pre_merger", "post_merger")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_result(path: Path, role: str) -> tuple[Path, dict, dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing {role} probe result: {path}")
    result = json.loads(path.read_text())
    if result.get("schema_version") != 1 or result.get("model", {}).get("model_role") != role:
        raise SystemExit(f"Probe result identity mismatch for {role}: {path}")
    cache_record = result.get("cache_manifest", {})
    cache_path = Path(cache_record.get("path", "")).expanduser().resolve()
    if not cache_path.is_file() or sha256_file(cache_path) != cache_record.get("sha256"):
        raise SystemExit(f"Probe cache manifest is missing or changed for {role}: {cache_path}")
    cache = json.loads(cache_path.read_text())
    if cache.get("schema_version") != 1 or cache.get("model", {}).get("model_role") != role:
        raise SystemExit(f"Feature cache identity mismatch for {role}: {cache_path}")
    if set(result.get("results", {})) != set(LEVELS):
        raise SystemExit(f"Probe result lacks feature levels for {role}: {path}")
    return path, result, cache


def protocol_signature(result: dict, cache: dict) -> dict:
    dataset = cache["dataset"]
    extraction = cache["extraction"]
    return {
        "class_list_sha256": dataset["class_list_sha256"],
        "labels_json_sha256": dataset["labels_json_sha256"],
        "splits": dataset["splits"],
        "extraction": {
            key: extraction[key]
            for key in (
                "attention_implementation",
                "class_count",
                "feature_dims",
                "feature_levels",
                "image_transform",
                "pooling",
                "seed",
                "selected_classes",
                "train_per_class",
                "val_per_class",
            )
        },
        "fit": result["config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in ROLES:
        parser.add_argument(f"--{role}-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    loaded = {role: load_result(getattr(args, f"{role}_result"), role) for role in ROLES}
    signatures = {role: protocol_signature(result, cache) for role, (_, result, cache) in loaded.items()}
    for role in ROLES[1:]:
        if signatures[role] != signatures["base"]:
            raise SystemExit(f"ImageNet-100 probe protocol mismatch for {role}")
    signature = signatures["base"]
    if (
        signature["extraction"]["class_count"] != 100
        or signature["extraction"]["train_per_class"] != 1300
        or signature["extraction"]["val_per_class"] != 50
    ):
        raise SystemExit("Refusing to aggregate a subset/smoke probe as the full ImageNet-100 result")

    rows = []
    summary = {}
    for level in LEVELS:
        scores = {role: float(loaded[role][1]["results"][level]["mean_top1"]) * 100.0 for role in ROLES}
        row = {"Feature": level, **{DISPLAY[role]: f"{scores[role]:.2f}" for role in ROLES}}
        row.update(
            {
                f"{DISPLAY[role]}-Base": f"{scores[role] - scores['base']:+.2f}"
                for role in ROLES[1:]
            }
        )
        rows.append(row)
        summary[level] = {
            **scores,
            **{f"{role}_minus_base": scores[role] - scores["base"] for role in ROLES[1:]},
        }
    bundle = {
        "inputs": {
            role: {"path": str(path), "sha256": sha256_file(path)}
            for role, (path, _result, _cache) in loaded.items()
        },
        "protocol": signature,
        "results_percent": summary,
        "schema_version": 1,
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse probe-summary directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        with (staging / "imagenet100_probe_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (staging / "imagenet100_probe_results.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, output_dir)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
