#!/usr/bin/env python3
"""Build deterministic 60/40 MMBench Golden-RLVR Parquet files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DATASET_NAME = "MMBench_GoldenQA"
DATA_SOURCE = f"datasets/{DATASET_NAME}"
SOURCE_NAME = "MMBench_DEV_EN"
SOURCE_MD5 = "b6caf1133a01c6bb705cf753bb527ed8"
SOURCE_ROWS = 4_329
SEED = 42
HELDOUT_FRACTION = 0.4
EXPECTED_L2 = {
    "logic_reasoning",
    "relation_reasoning",
    "attribute_reasoning",
    "coarse_perception",
    "finegrained_perception (cross-instance)",
    "finegrained_perception (instance-level)",
}
EXPECTED_OUTPUT_ROWS = {"train.parquet": 2_607, "test.parquet": 1_722}


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, category: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\0{SOURCE_NAME}\0{category}\0{key}".encode()).hexdigest()


def index_digest(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def split_indices(frame, seed: int = SEED) -> tuple[list[int], list[int], dict]:
    required = {"index", "l2-category"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"MMBench lacks required columns: {sorted(required - set(frame.columns))}")
    if len(frame) != SOURCE_ROWS or set(frame["l2-category"].astype(str)) != EXPECTED_L2:
        raise SystemExit("MMBench source is not the checksum-pinned old DEV-EN protocol")
    numeric = frame["index"].astype(int)
    if numeric.duplicated().any():
        raise SystemExit("MMBench source contains duplicate indices")
    work = frame.copy()
    work["_split_key"] = (numeric % 1_000_000).astype(str)
    strata: dict[str, list[str]] = defaultdict(list)
    group_sizes = Counter()
    for key, group in work.groupby("_split_key", sort=False):
        if len(group) not in (2, 3, 4):
            raise SystemExit(f"MMBench circular group {key} has unsupported size {len(group)}")
        categories = set(group["l2-category"].astype(str))
        if len(categories) != 1:
            raise SystemExit(f"MMBench circular group {key} crosses categories: {sorted(categories)}")
        strata[next(iter(categories))].append(str(key))
        group_sizes[len(group)] += 1
    heldout_keys: set[str] = set()
    strata_counts = {}
    for category in sorted(strata):
        keys = sorted(strata[category], key=lambda key: (stable_rank(seed, category, key), key))
        heldout_count = math.floor(len(keys) * HELDOUT_FRACTION + 0.5)
        selected = keys[:heldout_count]
        heldout_keys.update(selected)
        strata_counts[category] = {
            "heldout_groups": len(selected),
            "total_groups": len(keys),
            "train_groups": len(keys) - len(selected),
        }
    heldout_mask = work["_split_key"].isin(heldout_keys)
    train_positions = frame.index[~heldout_mask].tolist()
    test_positions = frame.index[heldout_mask].tolist()
    train_indices = [str(value) for value in frame.loc[train_positions, "index"]]
    test_indices = [str(value) for value in frame.loc[test_positions, "index"]]
    if len(train_positions) != EXPECTED_OUTPUT_ROWS["train.parquet"] or len(test_positions) != EXPECTED_OUTPUT_ROWS["test.parquet"]:
        raise SystemExit(
            f"Unexpected Golden split rows: train={len(train_positions)}, test={len(test_positions)}"
        )
    return train_positions, test_positions, {
        "group_key": "integer(index) modulo 1000000",
        "group_size_histogram": {str(key): value for key, value in sorted(group_sizes.items())},
        "heldout_fraction_actual": len(test_positions) / len(frame),
        "heldout_group_keys": sorted(heldout_keys, key=int),
        "method": "l2-category-stratified deterministic SHA256 ordering of complete circular groups",
        "seed": seed,
        "strata": strata_counts,
        "test_indices": test_indices,
        "test_indices_sha256": index_digest(test_indices),
        "train_indices": train_indices,
        "train_indices_sha256": index_digest(train_indices),
    }


def decode_image(value) -> bytes:
    if not isinstance(value, str) or not value:
        raise SystemExit("MMBench image field is empty or non-string")
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise SystemExit("MMBench image field is not valid Base64") from exc


def resolve_image_bytes(frame) -> dict[int, bytes]:
    raw_by_index = {str(int(row["index"])): row["image"] for _, row in frame.iterrows()}
    resolved: dict[int, bytes] = {}
    resolving: set[str] = set()

    def resolve(index: str) -> bytes:
        numeric_index = int(index)
        if numeric_index in resolved:
            return resolved[numeric_index]
        if index in resolving:
            raise SystemExit(f"Cyclic MMBench image reference at index {index}")
        if index not in raw_by_index:
            raise SystemExit(f"MMBench image reference points to missing index {index}")
        resolving.add(index)
        value = raw_by_index[index]
        if isinstance(value, str) and value.isdigit() and value in raw_by_index and value != index:
            image_bytes = resolve(value)
        else:
            image_bytes = decode_image(value)
        resolving.remove(index)
        resolved[numeric_index] = image_bytes
        return image_bytes

    for index in raw_by_index:
        resolve(index)
    return resolved


def prompt_for_row(row) -> str:
    import pandas as pd

    options = []
    for letter in "ABCD":
        value = row.get(letter)
        if value is not None and not pd.isna(value):
            options.append(f"{letter}. {value}")
    answer = str(row["answer"]).strip()
    if answer not in {item[0] for item in options}:
        raise SystemExit(f"MMBench answer {answer!r} is not among available options")
    hint = row.get("hint")
    hint_text = "" if hint is None or pd.isna(hint) else f"Hint: {hint}\n"
    return (
        "<image>\n\n"
        f"{hint_text}Question: {row['question']}\n"
        "Options:\n"
        + "\n".join(options)
        + "\n\nThink step by step and then provide the final option letter. "
        "Your response must end with <answer>LETTER</answer>, where LETTER is one of the available options."
    )


def build_records(frame, positions: list[int], split: str, images: dict[int, bytes]) -> list[dict]:
    records = []
    for position in positions:
        row = frame.loc[position]
        answer = str(row["answer"]).strip()
        prompt = prompt_for_row(row)
        records.append(
            {
                "index": int(row["index"]),
                "images": [{"bytes": images[int(row["index"])], "path": None}],
                "data_source": DATA_SOURCE,
                "prompt": [{"content": prompt, "role": "user"}],
                "ability": "vqa",
                "reward_model": {"ground_truth": answer, "style": "rule"},
                "extra_info": {
                    "answer": answer,
                    "category": str(row["l2-category"]),
                    "index": int(row["index"]),
                    "question": str(row["question"]),
                    "split": split,
                },
            }
        )
    return records


def arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field(
                "images",
                pa.list_(pa.field("element", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]))),
            ),
            pa.field("data_source", pa.string()),
            pa.field(
                "prompt",
                pa.list_(pa.field("element", pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())]))),
            ),
            pa.field("ability", pa.string()),
            pa.field("reward_model", pa.struct([pa.field("ground_truth", pa.string()), pa.field("style", pa.string())])),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        pa.field("answer", pa.string()),
                        pa.field("category", pa.string()),
                        pa.field("index", pa.int64()),
                        pa.field("question", pa.string()),
                        pa.field("split", pa.string()),
                    ]
                ),
            ),
        ]
    )


def huggingface_features():
    import datasets

    return datasets.Features(
        {
            "index": datasets.Value("int64"),
            "images": datasets.List(datasets.Image()),
            "data_source": datasets.Value("string"),
            "prompt": datasets.List(
                {"content": datasets.Value("string"), "role": datasets.Value("string")}
            ),
            "ability": datasets.Value("string"),
            "reward_model": {
                "ground_truth": datasets.Value("string"),
                "style": datasets.Value("string"),
            },
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
    if parquet.metadata.num_rows != expected_rows or not parquet.schema_arrow.remove_metadata().equals(arrow_schema()):
        raise SystemExit(f"Golden parquet row/schema mismatch: {path}")
    metadata = parquet.schema_arrow.metadata or {}
    if b'huggingface' not in metadata or '"_type": "Image"' not in metadata[b'huggingface'].decode():
        raise SystemExit(f"Golden parquet lacks Hugging Face Image feature metadata: {path}")
    table = pq.read_table(path, columns=["data_source", "reward_model", "extra_info"])
    if set(table["data_source"].to_pylist()) != {DATA_SOURCE}:
        raise SystemExit(f"Golden parquet data_source mismatch: {path}")
    rewards = table["reward_model"].to_pylist()
    if any(record.get("style") != "rule" or record.get("ground_truth") not in "ABCD" for record in rewards):
        raise SystemExit(f"Golden parquet reward schema mismatch: {path}")
    if {record.get("split") for record in table["extra_info"].to_pylist()} != {expected_split}:
        raise SystemExit(f"Golden parquet logical split mismatch: {path}")
    return {"bytes": path.stat().st_size, "rows": expected_rows, "sha256": digest_file(path, "sha256")}


def validate_existing(output_dir: Path, source_tsv: Path) -> dict | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Refusing to reuse Golden data without manifest: {output_dir}")
    manifest = json.loads(manifest_path.read_text())
    expected_protocol = {
        "heldout_fraction": HELDOUT_FRACTION,
        "schema_version": 1,
        "seed": SEED,
        "source_md5": SOURCE_MD5,
    }
    if manifest.get("protocol") != expected_protocol:
        raise SystemExit(f"Existing Golden protocol mismatch: {manifest_path}")
    if manifest.get("generator_script_sha256") != digest_file(Path(__file__), "sha256"):
        raise SystemExit(f"Existing Golden generator identity mismatch: {manifest_path}")
    if manifest.get("source", {}).get("sha256") != digest_file(source_tsv, "sha256"):
        raise SystemExit(f"Existing Golden source identity mismatch: {manifest_path}")
    for filename, split in (("train.parquet", "train"), ("test.parquet", "test")):
        record = validate_parquet(output_dir / filename, EXPECTED_OUTPUT_ROWS[filename], split)
        if record != manifest.get("outputs", {}).get(filename):
            raise SystemExit(f"Existing Golden output changed: {output_dir / filename}")
    return manifest


def published_summary(output_dir: Path, manifest: dict) -> dict:
    split = manifest["split"]
    return {
        "dataset_name": manifest["dataset_name"],
        "heldout_fraction_actual": split["heldout_fraction_actual"],
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
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pandas and PyArrow are required to prepare Golden data") from exc

    source_tsv = args.source_tsv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_tsv.is_file() or digest_file(source_tsv, "md5") != SOURCE_MD5:
        raise SystemExit(f"Expected checksum-pinned {SOURCE_NAME}: {source_tsv}")
    existing = validate_existing(output_dir, source_tsv)
    if existing is not None:
        print(json.dumps(published_summary(output_dir, existing), indent=2, sort_keys=True))
        return
    if args.verify_only:
        raise SystemExit(f"Golden dataset has not been generated: {output_dir}")

    frame = pd.read_csv(source_tsv, sep="\t")
    train_positions, test_positions, split_info = split_indices(frame)
    images = resolve_image_bytes(frame)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for filename, positions, split in (
            ("train.parquet", train_positions, "train"),
            ("test.parquet", test_positions, "test"),
        ):
            dataset = datasets.Dataset.from_list(
                build_records(frame, positions, split, images),
                features=huggingface_features(),
            )
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
            "protocol": {
                "heldout_fraction": HELDOUT_FRACTION,
                "schema_version": 1,
                "seed": SEED,
                "source_md5": SOURCE_MD5,
            },
            "source": {
                "bytes": source_tsv.stat().st_size,
                "md5": SOURCE_MD5,
                "path": str(source_tsv),
                "sha256": digest_file(source_tsv, "sha256"),
            },
            "split": split_info,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output_dir)
    print(json.dumps(published_summary(output_dir, manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
