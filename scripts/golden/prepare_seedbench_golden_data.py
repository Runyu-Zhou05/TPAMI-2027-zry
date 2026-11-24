#!/usr/bin/env python3
"""Build deterministic 60/40 SEED-Bench Golden-RLVR Parquet files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DATASET_NAME = "SEEDBench_GoldenQA"
DATA_SOURCE = f"datasets/{DATASET_NAME}"
SOURCE_NAME = "SEEDBench_IMG"
SOURCE_MD5 = "68017231464752261a2526d6ca3a10c0"
SOURCE_ROWS = 14_232
SEED = 42
HELDOUT_FRACTION = 0.4
EXPECTED_CATEGORIES = {
    "Text Understanding",
    "Visual Reasoning",
    "Scene Understanding",
    "Instance Identity",
    "Instance Interaction",
    "Instance Attributes",
    "Instance Location",
    "Spatial Relation",
    "Instances Counting",
}
EXPECTED_OUTPUT_ROWS = {"train.parquet": 8_539, "test.parquet": 5_693}


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, category: str, index: str) -> str:
    return hashlib.sha256(f"{seed}\0{SOURCE_NAME}\0{category}\0{index}".encode()).hexdigest()


def index_digest(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def split_indices(frame, seed: int = SEED) -> tuple[list[int], list[int], dict]:
    required = {"index", "category"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"SEED-Bench lacks required columns: {sorted(required - set(frame.columns))}")
    if len(frame) != SOURCE_ROWS or set(frame["category"].astype(str)) != EXPECTED_CATEGORIES:
        raise SystemExit("SEED-Bench source row/category identity mismatch")
    indices = frame["index"].astype(str).tolist()
    if len(indices) != len(set(indices)):
        raise SystemExit("SEED-Bench contains duplicate indices")
    strata: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for position, (index, category) in enumerate(zip(indices, frame["category"].astype(str), strict=True)):
        strata[category].append((position, index))
    heldout_positions: set[int] = set()
    strata_counts = {}
    for category in sorted(strata):
        rows = sorted(strata[category], key=lambda item: (stable_rank(seed, category, item[1]), item[1]))
        heldout_count = math.floor(len(rows) * HELDOUT_FRACTION + 0.5)
        heldout_positions.update(position for position, _ in rows[:heldout_count])
        strata_counts[category] = {
            "heldout": heldout_count,
            "total": len(rows),
            "train": len(rows) - heldout_count,
        }
    train_positions = [position for position in range(len(frame)) if position not in heldout_positions]
    test_positions = [position for position in range(len(frame)) if position in heldout_positions]
    train_indices = [indices[position] for position in train_positions]
    test_indices = [indices[position] for position in test_positions]
    if len(train_positions) != EXPECTED_OUTPUT_ROWS["train.parquet"] or len(test_positions) != EXPECTED_OUTPUT_ROWS["test.parquet"]:
        raise SystemExit(f"Unexpected SEED Golden split rows: train={len(train_positions)}, test={len(test_positions)}")
    return train_positions, test_positions, {
        "group_key": "individual row index",
        "heldout_fraction_actual": len(test_positions) / len(frame),
        "method": "category-stratified deterministic SHA256 ordering of rows",
        "seed": seed,
        "strata": strata_counts,
        "test_indices": test_indices,
        "test_indices_sha256": index_digest(test_indices),
        "train_indices": train_indices,
        "train_indices_sha256": index_digest(train_indices),
    }


def decode_image(value) -> bytes:
    if not isinstance(value, str) or not value:
        raise SystemExit("SEED-Bench image field is empty or non-string")
    try:
        image = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise SystemExit("SEED-Bench image field is not valid Base64") from exc
    if not image.startswith(b"\xff\xd8\xff"):
        raise SystemExit("SEED-Bench image is not a JPEG payload")
    return image


def prompt_for_row(row) -> str:
    options = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
    answer = str(row["answer"]).strip().upper()
    if answer not in "ABCD":
        raise SystemExit(f"Invalid SEED-Bench answer: {answer!r}")
    return (
        "<image>\n\n"
        f"Question: {row['question']}\nOptions:\n{options}\n\n"
        "Think step by step and then provide the final option letter. "
        "Your response must end with <answer>LETTER</answer>, where LETTER is one of A, B, C, or D."
    )


def build_records(frame, positions: list[int], split: str) -> list[dict]:
    records = []
    for position in positions:
        row = frame.iloc[position]
        answer = str(row["answer"]).strip().upper()
        records.append(
            {
                "index": int(row["index"]),
                "images": [{"bytes": decode_image(row["image"]), "path": None}],
                "data_source": DATA_SOURCE,
                "prompt": [{"content": prompt_for_row(row), "role": "user"}],
                "ability": "vqa",
                "reward_model": {"ground_truth": answer, "style": "rule"},
                "extra_info": {
                    "answer": answer,
                    "category": str(row["category"]),
                    "index": int(row["index"]),
                    "question": str(row["question"]),
                    "split": split,
                },
            }
        )
    return records


def huggingface_features():
    import datasets

    return datasets.Features(
        {
            "index": datasets.Value("int64"),
            "images": datasets.List(datasets.Image()),
            "data_source": datasets.Value("string"),
            "prompt": datasets.List({"content": datasets.Value("string"), "role": datasets.Value("string")}),
            "ability": datasets.Value("string"),
            "reward_model": {"ground_truth": datasets.Value("string"), "style": datasets.Value("string")},
            "extra_info": {
                "answer": datasets.Value("string"),
                "category": datasets.Value("string"),
                "index": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "split": datasets.Value("string"),
            },
        }
    )


def validate_parquet(path: Path, expected_rows: int, expected_split: str) -> dict:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    if parquet.metadata.num_rows != expected_rows or b"huggingface" not in metadata or '"_type": "Image"' not in metadata[b"huggingface"].decode():
        raise SystemExit(f"SEED Golden parquet row/image metadata mismatch: {path}")
    table = pq.read_table(path, columns=["index", "data_source", "reward_model", "extra_info"])
    if len(set(table["index"].to_pylist())) != expected_rows or set(table["data_source"].to_pylist()) != {DATA_SOURCE}:
        raise SystemExit(f"SEED Golden parquet identity mismatch: {path}")
    rewards = table["reward_model"].to_pylist()
    if any(record.get("style") != "rule" or record.get("ground_truth") not in "ABCD" for record in rewards):
        raise SystemExit(f"SEED Golden reward schema mismatch: {path}")
    if {record.get("split") for record in table["extra_info"].to_pylist()} != {expected_split}:
        raise SystemExit(f"SEED Golden logical split mismatch: {path}")
    return {"bytes": path.stat().st_size, "rows": expected_rows, "sha256": digest_file(path, "sha256")}


def validate_existing(output_dir: Path, source_tsv: Path) -> dict | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Refusing to reuse SEED Golden data without manifest: {output_dir}")
    manifest = json.loads(manifest_path.read_text())
    expected_protocol = {"heldout_fraction": 0.4, "schema_version": 1, "seed": 42, "source_md5": SOURCE_MD5}
    if (
        manifest.get("protocol") != expected_protocol
        or manifest.get("generator_script_sha256") != digest_file(Path(__file__), "sha256")
        or manifest.get("source", {}).get("sha256") != digest_file(source_tsv, "sha256")
    ):
        raise SystemExit(f"Existing SEED Golden identity mismatch: {manifest_path}")
    for filename, expected_rows in EXPECTED_OUTPUT_ROWS.items():
        split = "train" if filename.startswith("train") else "test"
        actual = validate_parquet(output_dir / filename, expected_rows, split)
        if actual != manifest.get("outputs", {}).get(filename):
            raise SystemExit(f"Existing SEED Golden output changed: {output_dir / filename}")
    return manifest


def published_summary(output_dir: Path, manifest: dict) -> dict:
    return {
        "dataset_name": manifest["dataset_name"],
        "heldout_fraction_actual": manifest["split"]["heldout_fraction_actual"],
        "manifest": str((output_dir / "manifest.json").resolve()),
        "outputs": manifest["outputs"],
        "seed": manifest["protocol"]["seed"],
        "status": "ready",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        import datasets
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("datasets, pandas, and PyArrow are required to prepare SEED Golden data") from exc
    source_tsv = args.source_tsv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_tsv.is_file() or digest_file(source_tsv, "md5") != SOURCE_MD5:
        raise SystemExit(f"Expected checksum-pinned {SOURCE_NAME}: {source_tsv}")
    existing = validate_existing(output_dir, source_tsv)
    if existing is not None:
        print(json.dumps(published_summary(output_dir, existing), indent=2, sort_keys=True))
        return
    if args.verify_only:
        raise SystemExit(f"SEED Golden dataset has not been generated: {output_dir}")
    frame = pd.read_csv(source_tsv, sep="\t")
    train_positions, test_positions, split_info = split_indices(frame)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for filename, positions, split in (("train.parquet", train_positions, "train"), ("test.parquet", test_positions, "test")):
            dataset = datasets.Dataset.from_list(build_records(frame, positions, split), features=huggingface_features())
            dataset.to_parquet(staging / filename, compression="zstd")
        outputs = {
            filename: validate_parquet(staging / filename, rows, "train" if filename.startswith("train") else "test")
            for filename, rows in EXPECTED_OUTPUT_ROWS.items()
        }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": DATASET_NAME,
            "generator_script_sha256": digest_file(Path(__file__), "sha256"),
            "outputs": outputs,
            "protocol": {"heldout_fraction": 0.4, "schema_version": 1, "seed": 42, "source_md5": SOURCE_MD5},
            "source": {"bytes": source_tsv.stat().st_size, "md5": SOURCE_MD5, "path": str(source_tsv), "sha256": digest_file(source_tsv, "sha256")},
            "split": split_info,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output_dir)
    print(json.dumps(published_summary(output_dir, manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
