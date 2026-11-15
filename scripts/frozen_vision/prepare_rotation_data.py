#!/usr/bin/env python3
"""Download pinned MMBench data and build the Rotation FVE Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fve_runtime import DEFAULT_OUTPUT_ROOT, ROOT, resolve_path


SOURCE_DATASET = "HuggingFaceM4/MMBench"
SOURCE_REVISION = "215b79e126bea0777a489683f239edd0a6012a8a"
DATASET_NAME = "MMBench_RotationQA_90degree"
DATA_SOURCE_LABEL = f"datasets/{DATASET_NAME}"
SEED = 42
GRANULARITY = 90
EXPECTED_ROWS = {"train.parquet": 6666, "test.parquet": 4329}
REFERENCE_SHA256 = {
    "train.parquet": "fb4a6e809f92d5c41f2d47f348a8c157c206c489a3b28cba3ddaca750fbc17a5",
    "test.parquet": "417ed302a203d3b5c51caa8e3f424ebb0969787ea89eb68e9b5796d94b63ca01",
}
RAW_SHA256 = {
    "data/test-00000-of-00001-147b3b3e74778350.parquet": (
        "eaca73c0cc6a2fae7efea8981f2ed04b29b5cb174ac5ff94a5651bc71cd6e6ce"
    ),
    "data/validation-00000-of-00001-93861598ee8837c0.parquet": (
        "741977ca4e9badf4383a244815c03b0de2b129aac4c97b0ff71784a0c63fa14f"
    ),
}
REQUIRED_COLUMNS = [
    "index",
    "source",
    "split",
    "height",
    "width",
    "images",
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
]
QUERY_TEMPLATE = f"""<image><image>

These are two images. The second image is a rotated version of the first image.\x20
Please determine how many degrees the second image has been rotated **counter-clockwise** relative to the first image.
The granuality of rotation is {GRANULARITY} degrees, meaning the second image could have been rotated by 0, {GRANULARITY}, {2 * GRANULARITY}, ..., {360 - GRANULARITY} degrees.

You must reason step-by-step and then provide the final answer.

The output **must strictly follow** this format:
<think> your reasoning here </think> <answer>number_of_degrees</answer>"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def rotated_rect_with_max_area(width: int, height: int, angle: float) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 0, 0
    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        half_short = 0.5 * side_short
        result = (
            (half_short / sin_a, half_short / cos_a)
            if width_is_longer
            else (half_short / cos_a, half_short / sin_a)
        )
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        result = (
            (width * cos_a - height * sin_a) / cos_2a,
            (height * cos_a - width * sin_a) / cos_2a,
        )
    return int(result[0]), int(result[1])


def validate_dataset(dataset_dir: Path, expect_complete: bool = True) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("PyArrow is required; build the environment described in AGENTS.md first") from exc

    result = {}
    for filename in ("train.parquet", "test.parquet"):
        path = dataset_dir / filename
        if not path.is_file():
            raise SystemExit(f"Missing processed Rotation dataset file: {path}")
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise SystemExit(f"{path} is missing required columns: {missing}")
        rows = parquet.metadata.num_rows
        if expect_complete and rows != EXPECTED_ROWS[filename]:
            raise SystemExit(f"{path} has {rows} rows; expected {EXPECTED_ROWS[filename]}")
        first_source = pq.read_table(path, columns=["data_source"]).column(0)[0].as_py()
        if "RotationQA" not in first_source:
            raise SystemExit(f"Unexpected data_source in {path}: {first_source!r}")
        actual_sha256 = sha256_file(path)
        result[filename] = {
            "bytes": path.stat().st_size,
            "rows": rows,
            "columns": columns,
            "sha256": actual_sha256,
            "reference_sha256_247": REFERENCE_SHA256[filename],
            "matches_reference_247": actual_sha256 == REFERENCE_SHA256[filename],
        }
    return result


def validate_existing(dataset_dir: Path, expect_complete: bool) -> dict | None:
    manifest_path = dataset_dir / "manifest.json"
    if not dataset_dir.exists():
        return None
    if not manifest_path.is_file():
        raise SystemExit(
            f"Refusing to trust existing data without a manifest: {dataset_dir}. "
            "Move it aside or choose a new --data-root."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_dataset") != SOURCE_DATASET or manifest.get("source_revision") != SOURCE_REVISION:
        raise SystemExit(f"Existing manifest does not match the pinned MMBench source: {manifest_path}")
    files = validate_dataset(dataset_dir, expect_complete=expect_complete)
    for filename, metadata in files.items():
        recorded = manifest.get("files", {}).get(filename, {}).get("sha256")
        if recorded != metadata["sha256"]:
            raise SystemExit(f"Checksum mismatch against {manifest_path}: {filename}")
    print(f"Validated existing Rotation dataset: {dataset_dir}")
    print(json.dumps(files, indent=2, sort_keys=True))
    return files


def locate_raw_files(raw_dir: Path) -> dict[str, list[Path]]:
    split_files = {
        split: sorted((raw_dir / "data").glob(f"{split}-*.parquet"))
        for split in ("test", "validation")
    }
    for split, files in split_files.items():
        if not files:
            raise SystemExit(f"MMBench snapshot has no {split} Parquet files under {raw_dir / 'data'}")
    located = {str(path.relative_to(raw_dir)): path for paths in split_files.values() for path in paths}
    if set(located) != set(RAW_SHA256):
        raise SystemExit(
            f"Raw MMBench file set does not match pinned revision {SOURCE_REVISION}: {sorted(located)}"
        )
    for relative_path, expected_sha256 in RAW_SHA256.items():
        actual_sha256 = sha256_file(located[relative_path])
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"Raw MMBench checksum mismatch for {located[relative_path]}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    return split_files


def download_raw_snapshot(raw_dir: Path, cache_dir: Path) -> dict[str, list[Path]]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required; build the environment described in AGENTS.md first") from exc

    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=SOURCE_DATASET,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        local_dir=raw_dir,
        cache_dir=cache_dir,
        allow_patterns=["README.md", "data/*.parquet"],
    )
    return locate_raw_files(raw_dir)


def build_split(
    source_files: list[Path],
    source_split: str,
    logical_split: str,
    output_path: Path,
    cache_dir: Path,
    num_proc: int,
    max_rows: int | None,
) -> None:
    try:
        import datasets
    except ImportError as exc:
        raise SystemExit("datasets is required; build the environment described in AGENTS.md first") from exc

    sys.path.insert(0, str(ROOT))
    from build_benchmarks.image_utils import decode_image

    raw = datasets.load_dataset(
        "parquet",
        data_files={source_split: [str(path) for path in source_files]},
        split=source_split,
        cache_dir=str(cache_dir),
    )
    if max_rows is not None:
        raw = raw.select(range(min(max_rows, len(raw))))

    rng = random.Random(SEED)
    angles = [rng.choice(range(0, 360, GRANULARITY)) for _ in range(len(raw))]

    def transform(example, index):
        image = decode_image(example["image"])
        angle = angles[index]
        rotated = image.rotate(angle, expand=True)
        rotated_width, rotated_height = rotated_rect_with_max_area(
            image.width, image.height, math.radians(angle)
        )
        rotated = rotated.crop(
            (
                (rotated.width - rotated_width) / 2,
                (rotated.height - rotated_height) / 2,
                (rotated.width + rotated_width) / 2,
                (rotated.height + rotated_height) / 2,
            )
        )
        if rotated.width < 28 or rotated.height < 28:
            raise ValueError(f"Rotation produced an undersized image at {source_split}[{index}]")
        answer = str(angle)
        return {
            "index": example["index"],
            "source": example["source"],
            "split": example["split"],
            "height": rotated.height,
            "width": rotated.width,
            "images": [image, rotated],
            "data_source": DATA_SOURCE_LABEL,
            "prompt": [{"role": "user", "content": QUERY_TEMPLATE}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": logical_split,
                "index": index,
                "answer": answer,
                "question": QUERY_TEMPLATE,
            },
        }

    processed = raw.map(
        transform,
        with_indices=True,
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc=f"Building Rotation {logical_split}",
    )
    processed.to_parquet(output_path)
    print(f"Wrote {len(processed)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.environ.get("FVE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--data-root", default=os.environ.get("FVE_DATA_ROOT"))
    parser.add_argument("--raw-root", default=os.environ.get("FVE_RAW_DATA_ROOT"))
    parser.add_argument(
        "--raw-snapshot",
        default=None,
        help="Use an existing pinned MMBench snapshot instead of downloading it",
    )
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Development-only partial build; writes to a suffixed directory and is not valid for training",
    )
    args = parser.parse_args()
    if args.num_proc < 1:
        raise SystemExit("--num-proc must be at least 1")
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        raise SystemExit("--max-rows-per-split must be positive")

    output_root = resolve_path(args.output_root)
    data_root = resolve_path(args.data_root or output_root / "inputs" / "datasets")
    raw_root = resolve_path(args.raw_root or output_root / "inputs" / "raw")
    cache_dir = output_root / "cache" / "huggingface"
    directory_name = DATASET_NAME
    if args.max_rows_per_split is not None:
        directory_name += f"_partial_{args.max_rows_per_split}"
    dataset_dir = data_root / directory_name
    expect_complete = args.max_rows_per_split is None

    existing = validate_existing(dataset_dir, expect_complete=expect_complete)
    if existing is not None:
        return
    if args.verify_only:
        raise SystemExit(f"No prepared Rotation dataset found at {dataset_dir}")

    if args.raw_snapshot is not None:
        raw_dir = resolve_path(args.raw_snapshot)
        split_files = locate_raw_files(raw_dir)
    else:
        raw_dir = raw_root / f"HuggingFaceM4--MMBench--{SOURCE_REVISION}"
        split_files = download_raw_snapshot(raw_dir, cache_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory_name}.building-", dir=data_root) as temporary:
        staging = Path(temporary)
        build_split(
            split_files["test"],
            source_split="test",
            logical_split="train",
            output_path=staging / "train.parquet",
            cache_dir=cache_dir,
            num_proc=args.num_proc,
            max_rows=args.max_rows_per_split,
        )
        build_split(
            split_files["validation"],
            source_split="validation",
            logical_split="test",
            output_path=staging / "test.parquet",
            cache_dir=cache_dir,
            num_proc=args.num_proc,
            max_rows=args.max_rows_per_split,
        )
        files = validate_dataset(staging, expect_complete=expect_complete)
        raw_files = {
            str(path.relative_to(raw_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for paths in split_files.values()
            for path in paths
        }
        try:
            import datasets
            import PIL
            import pyarrow

            package_versions = {
                "datasets": datasets.__version__,
                "pillow": PIL.__version__,
                "pyarrow": pyarrow.__version__,
            }
        except ImportError:
            package_versions = {}
        manifest = {
            "complete": expect_complete,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_source_label": DATA_SOURCE_LABEL,
            "files": files,
            "generator_git_commit": git_commit(),
            "granularity_degrees": GRANULARITY,
            "max_rows_per_split": args.max_rows_per_split,
            "package_versions": package_versions,
            "raw_files": raw_files,
            "seed": SEED,
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "split_mapping": {"test": "train.parquet", "validation": "test.parquet"},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, dataset_dir)
    print(f"Rotation dataset is ready: {dataset_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
