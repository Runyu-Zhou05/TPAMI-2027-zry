#!/usr/bin/env python3
"""Download and verify MathVista/MMMU reasoning evaluation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {
    "MathVista_MINI": {
        "class": "MathVista",
        "md5": "f199b98e178e5a2a20e7048f5dcb0464",
        "url": "https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv",
    },
    "MMMU_DEV_VAL": {
        "class": "MMMUDataset",
        "md5": "585e8ad75e73f75dcad265dfd0417d64",
        "url": "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv",
    },
}


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_tsv(name: str, path: Path) -> dict:
    expected_md5 = DATASETS[name]["md5"]
    actual_md5 = digest(path, "md5")
    if actual_md5 != expected_md5:
        raise SystemExit(f"{name} MD5 mismatch: expected {expected_md5}, got {actual_md5}")
    with path.open(newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    required = {"index", "question", "answer"}
    if not required.issubset(header):
        raise SystemExit(f"{name} TSV lacks required columns: {sorted(required - set(header))}")
    return {
        "bytes": path.stat().st_size,
        "class": DATASETS[name]["class"],
        "columns": header,
        "md5": actual_md5,
        "path": str(path.resolve()),
        "sha256": digest(path, "sha256"),
        "url": DATASETS[name]["url"],
    }


def download_one(name: str, root: Path) -> dict:
    target = root / f"{name}.tsv"
    if not target.is_file() or digest(target, "md5") != DATASETS[name]["md5"]:
        curl = shutil.which("curl")
        if curl is None:
            raise SystemExit("curl is required for checksum-pinned OpenCompass downloads")
        temporary = target.with_suffix(".tsv.download")
        temporary.unlink(missing_ok=True)
        # OpenCompass currently serves an expired certificate. The official
        # pinned MD5 is verified before the downloaded bytes are published.
        subprocess.run(
            [
                curl,
                "--fail",
                "--insecure",
                "--location",
                "--retry",
                "3",
                "--output",
                str(temporary),
                DATASETS[name]["url"],
            ],
            check=True,
        )
        try:
            validate_tsv(name, temporary)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        os.replace(temporary, target)
    return validate_tsv(name, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmu-data-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = args.lmu_data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in DATASETS:
        path = root / f"{name}.tsv"
        if args.verify_only:
            if not path.is_file():
                raise SystemExit(f"Missing reasoning dataset: {path}")
            records[name] = validate_tsv(name, path)
        else:
            records[name] = download_one(name, root)
    manifest = {
        "benchmarks": records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "MathVista": "MathVista_MINI with explicit chain-of-thought generation and official auxiliary answer extraction",
            "MMMU": "MMMU_DEV_VAL with explicit chain-of-thought generation and exact-matching evaluation",
        },
        "schema_version": 1,
    }
    output = root / "reasoning_benchmark_manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
