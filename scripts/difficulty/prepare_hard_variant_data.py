#!/usr/bin/env python3
"""Convert generated Table-XI hard variants into immutable verl RL Parquets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TASKS = {
    "position3x3": {"dataset": "MMBench-PositionQA_3x3", "images": 2},
    "rotation45": {"dataset": "MMBench_RotationQA_45degree", "images": 2},
    "jigsaw3x3": {"dataset": "MMBench_JigsawQA_3x3", "images": 9},
    "contrastive_strong": {"dataset": "MMBench-ContrastiveQA_strong", "images": 2},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(root: Path) -> dict:
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not files:
        raise SystemExit(f"Generated hard-variant dataset has no Parquet files: {root}")
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    }


def validate_processed(path: Path, expected_images: int, data_source: str, split: str) -> dict:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows <= 0:
        raise SystemExit(f"Empty processed hard-variant data: {path}")
    metadata = parquet.schema_arrow.metadata or {}
    if b"huggingface" not in metadata or '"_type": "Image"' not in metadata[b"huggingface"].decode():
        raise SystemExit(f"Processed hard-variant data lacks HF Image metadata: {path}")
    required = {"data_source", "prompt", "images", "reward_model", "extra_info"}
    if not required.issubset(parquet.schema_arrow.names):
        raise SystemExit(f"Processed hard-variant schema is incomplete: {path}")
    table = pq.read_table(path, columns=list(required))
    if set(table["data_source"].to_pylist()) != {data_source}:
        raise SystemExit(f"Processed hard-variant data_source mismatch: {path}")
    images = table["images"].to_pylist()
    prompts = table["prompt"].to_pylist()
    if any(len(value) != expected_images for value in images):
        raise SystemExit(f"Processed hard-variant image count mismatch: {path}")
    if any(record[0]["content"].count("<image>") != expected_images for record in prompts):
        raise SystemExit(f"Processed hard-variant prompt/image placeholders mismatch: {path}")
    if {record.get("split") for record in table["extra_info"].to_pylist()} != {split}:
        raise SystemExit(f"Processed hard-variant logical split mismatch: {path}")
    return {"bytes": path.stat().st_size, "rows": parquet.metadata.num_rows, "sha256": sha256_file(path)}


def validate_existing(target: Path, task: str, source: dict) -> dict | None:
    if not target.exists():
        return None
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Refusing to reuse hard-variant data without manifest: {target}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("task") != task
        or manifest.get("source_files") != source
        or manifest.get("generator_script_sha256") != sha256_file(Path(__file__))
    ):
        raise SystemExit(f"Hard-variant processed data identity mismatch: {manifest_path}")
    spec = TASKS[task]
    data_source = f"datasets/{spec['dataset']}"
    for filename, split in (("train.parquet", "train"), ("test.parquet", "test")):
        actual = validate_processed(target / filename, spec["images"], data_source, split)
        if actual != manifest.get("outputs", {}).get(filename):
            raise SystemExit(f"Hard-variant processed output changed: {target / filename}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--raw-root", type=Path, required=True, help="Parent generated datasets directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Parent processed RL datasets directory")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.num_proc < 1:
        raise SystemExit("--num-proc must be positive")
    try:
        import datasets
    except ImportError as exc:
        raise SystemExit("datasets and PyArrow are required to preprocess hard variants") from exc

    spec = TASKS[args.task]
    raw_dir = (args.raw_root / spec["dataset"]).expanduser().resolve()
    target = (args.output_root / spec["dataset"]).expanduser().resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"Generated hard-variant directory is missing: {raw_dir}")
    inventory = source_inventory(raw_dir)
    existing = validate_existing(target, args.task, inventory)
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    if args.verify_only:
        raise SystemExit(f"Hard-variant processed data has not been generated: {target}")

    source = datasets.load_dataset(str(raw_dir))
    if set(source) != {"test", "validation"}:
        raise SystemExit(f"Expected generated test/validation splits, found {sorted(source)}")
    data_source = f"datasets/{spec['dataset']}"

    def transform(split: str):
        def apply(example, index):
            query = example["query"]
            answer = str(example["answer"])
            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": query}],
                "images": example["images"],
                "ability": "vision_ssl",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "answer": answer,
                    "question": query,
                },
            }

        return apply

    mapped = {
        "train": source["test"].map(
            transform("train"), with_indices=True, remove_columns=source["test"].column_names, num_proc=args.num_proc
        ),
        "test": source["validation"].map(
            transform("test"), with_indices=True, remove_columns=source["validation"].column_names, num_proc=args.num_proc
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.building-", dir=target.parent) as temporary:
        staging = Path(temporary)
        mapped["train"].to_parquet(staging / "train.parquet")
        mapped["test"].to_parquet(staging / "test.parquet")
        outputs = {
            "train.parquet": validate_processed(staging / "train.parquet", spec["images"], data_source, "train"),
            "test.parquet": validate_processed(staging / "test.parquet", spec["images"], data_source, "test"),
        }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": spec["dataset"],
            "expected_images": spec["images"],
            "generator_script_sha256": sha256_file(Path(__file__)),
            "outputs": outputs,
            "schema_version": 1,
            "source_files": inventory,
            "task": args.task,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, target)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
