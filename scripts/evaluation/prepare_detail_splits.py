#!/usr/bin/env python3
"""Create the deterministic 40% held-out MMBench/SEED detail split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from prepare_evaluation_data import DATASETS, digest


DATASET_NAMES = ("MMBench_DEV_EN_V11", "SEEDBench_IMG")
HELDOUT_FILENAMES = {
    "MMBench_DEV_EN_V11": "MMBench_DEV_EN_V11_FVE_HELDOUT.tsv",
    "SEEDBench_IMG": "SEEDBench_IMG_FVE_HELDOUT.tsv",
}
SCHEMA_VERSION = 1
DEFAULT_SEED = 42
HELDOUT_FRACTION = 0.4
MMBENCH_ROWS = 4_876
MMBENCH_L2_CATEGORIES = {
    "logic_reasoning",
    "relation_reasoning",
    "attribute_reasoning",
    "coarse_perception",
    "finegrained_perception (cross-instance)",
    "finegrained_perception (instance-level)",
}
SEED_ROWS = 14_232
SEED_CATEGORIES = {
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return digest(path, "sha256")


def dataframe_tsv_sha256(frame) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tsv") as handle:
        frame.to_csv(handle.name, sep="\t", index=False)
        return sha256_file(Path(handle.name))


def stable_rank(seed: int, dataset: str, stratum: str, key: str) -> str:
    return sha256_bytes(f"{seed}\0{dataset}\0{stratum}\0{key}".encode())


def heldout_count(size: int) -> int:
    # Explicit half-up rounding avoids Python's implicit bankers rounding.
    return math.floor(size * HELDOUT_FRACTION + 0.5)


def split_strata(items: dict[str, list[str]], seed: int, dataset: str) -> tuple[set[str], dict]:
    heldout: set[str] = set()
    counts = {}
    for stratum in sorted(items):
        keys = sorted(items[stratum], key=lambda key: (stable_rank(seed, dataset, stratum, key), key))
        selected = keys[: heldout_count(len(keys))]
        heldout.update(selected)
        counts[stratum] = {
            "heldout": len(selected),
            "total": len(keys),
            "train": len(keys) - len(selected),
        }
    return heldout, counts


def canonical_indices(values) -> list[str]:
    return [str(value) for value in values]


def index_digest(indices: list[str]) -> str:
    return sha256_bytes(("\n".join(indices) + "\n").encode())


def split_mmbench(frame, seed: int):
    required = {"index", "l2-category"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"MMBench lacks required columns: {sorted(required - set(frame.columns))}")
    if len(frame) != MMBENCH_ROWS or set(frame["l2-category"].astype(str)) != MMBENCH_L2_CATEGORIES:
        raise SystemExit("MMBench row count or six-category schema does not match DEV-EN v1.1")
    numeric = frame["index"].astype(int)
    if numeric.duplicated().any():
        raise SystemExit("MMBench contains duplicate indices")
    work = frame.copy()
    work["_split_key"] = (numeric % 1_000_000).astype(str)
    strata: dict[str, list[str]] = defaultdict(list)
    for key, group in work.groupby("_split_key", sort=False):
        if len(group) != 4:
            raise SystemExit(f"MMBench circular group {key} has {len(group)} rows instead of 4")
        categories = set(group["l2-category"].astype(str))
        if len(categories) != 1:
            raise SystemExit(f"MMBench circular group {key} crosses l2 categories: {sorted(categories)}")
        strata[next(iter(categories))].append(str(key))
    heldout_keys, strata_counts = split_strata(strata, seed, "MMBench_DEV_EN_V11")
    heldout_mask = work["_split_key"].isin(heldout_keys)
    heldout = frame.loc[heldout_mask].copy()
    train_indices = canonical_indices(frame.loc[~heldout_mask, "index"])
    heldout_indices = canonical_indices(frame.loc[heldout_mask, "index"])
    return heldout, {
        "group_key": "integer(index) modulo 1000000",
        "heldout_group_keys": sorted(heldout_keys, key=int),
        "heldout_indices": heldout_indices,
        "heldout_indices_sha256": index_digest(heldout_indices),
        "method": "stratified deterministic SHA256 ordering of complete four-row circular groups",
        "strata": strata_counts,
        "train_indices": train_indices,
        "train_indices_sha256": index_digest(train_indices),
    }


def split_seed(frame, seed: int):
    required = {"index", "category"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"SEEDBench lacks required columns: {sorted(required - set(frame.columns))}")
    if len(frame) != SEED_ROWS or set(frame["category"].astype(str)) != SEED_CATEGORIES:
        raise SystemExit("SEEDBench row count or nine-category schema does not match SEEDBench_IMG")
    indices = canonical_indices(frame["index"])
    if len(indices) != len(set(indices)):
        raise SystemExit("SEEDBench contains duplicate indices")
    strata: dict[str, list[str]] = defaultdict(list)
    for index, category in zip(indices, frame["category"].astype(str)):
        strata[category].append(index)
    heldout_keys, strata_counts = split_strata(strata, seed, "SEEDBench_IMG")
    heldout_mask = frame["index"].astype(str).isin(heldout_keys)
    heldout = frame.loc[heldout_mask].copy()
    train_indices = canonical_indices(frame.loc[~heldout_mask, "index"])
    heldout_indices = canonical_indices(frame.loc[heldout_mask, "index"])
    return heldout, {
        "group_key": "individual row index",
        "heldout_indices": heldout_indices,
        "heldout_indices_sha256": index_digest(heldout_indices),
        "method": "category-stratified deterministic SHA256 ordering of rows",
        "strata": strata_counts,
        "train_indices": train_indices,
        "train_indices_sha256": index_digest(train_indices),
    }


def validate_sources(lmu_root: Path) -> tuple[dict[str, Path], dict]:
    source_manifest_path = lmu_root / "six_benchmark_manifest.json"
    if not source_manifest_path.is_file():
        raise SystemExit(f"Missing evaluation-data manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    paths = {}
    sources = {}
    for name in DATASET_NAMES:
        path = (lmu_root / f"{name}.tsv").resolve()
        info = source_manifest.get("benchmarks", {}).get(name, {})
        if not path.is_file() or Path(info.get("path", "")).expanduser().resolve() != path:
            raise SystemExit(f"Missing canonical source data for {name}: {path}")
        md5 = digest(path, "md5")
        sha256 = digest(path, "sha256")
        if md5 != DATASETS[name]["md5"] or md5 != info.get("md5") or sha256 != info.get("sha256"):
            raise SystemExit(f"Source checksum mismatch for {name}: {path}")
        paths[name] = path
        sources[name] = {"bytes": path.stat().st_size, "md5": md5, "path": str(path), "sha256": sha256}
    return paths, {
        "manifest": str(source_manifest_path.resolve()),
        "manifest_sha256": sha256_file(source_manifest_path),
        "datasets": sources,
    }


def validate_existing(target: Path, expected_protocol: dict, sources: dict[str, Path]) -> dict | None:
    if not target.exists():
        return None
    manifest_path = target / "detail_split_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Refusing to reuse detail split without manifest: {target}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != expected_protocol:
        raise SystemExit(f"Existing detail split protocol does not match: {manifest_path}")
    if manifest.get("generator_script_sha256") != sha256_file(Path(__file__)):
        raise SystemExit(f"Existing detail split generator identity does not match: {manifest_path}")
    import pandas as pd

    partition_identity = {}
    for name in DATASET_NAMES:
        entry = manifest.get("datasets", {}).get(name, {})
        path = Path(entry.get("heldout_tsv", "")).expanduser().resolve()
        expected = (target / "heldout" / HELDOUT_FILENAMES[name]).resolve()
        if path != expected or not path.is_file() or sha256_file(path) != entry.get("heldout_tsv_sha256"):
            raise SystemExit(f"Existing held-out TSV is missing or changed: {expected}")
        source_frame = pd.read_csv(sources[name], sep="\t")
        if name == "MMBench_DEV_EN_V11":
            expected_heldout, expected_info = split_mmbench(source_frame, int(expected_protocol["seed"]))
        else:
            expected_heldout, expected_info = split_seed(source_frame, int(expected_protocol["seed"]))
        actual = pd.read_csv(path, sep="\t")
        actual_indices = canonical_indices(actual["index"])
        expected_indices = canonical_indices(expected_heldout["index"])
        if actual_indices != expected_indices or entry.get("heldout_indices") != expected_info["heldout_indices"]:
            raise SystemExit(f"Existing held-out partition is not the declared deterministic split: {path}")
        if sha256_file(path) != dataframe_tsv_sha256(expected_heldout):
            raise SystemExit(f"Existing held-out rows differ from the pinned source partition: {path}")
        for key in ("heldout_indices_sha256", "train_indices_sha256"):
            if entry.get(key) != expected_info[key]:
                raise SystemExit(f"Existing detail split {key} mismatch for {name}")
        partition_identity[name] = {
            "heldout_indices_sha256": expected_info["heldout_indices_sha256"],
            "train_indices_sha256": expected_info["train_indices_sha256"],
        }
    expected_split_id = sha256_bytes(
        json.dumps(
            {"partition": partition_identity, "protocol": expected_protocol},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    if manifest.get("split_id") != expected_split_id:
        raise SystemExit(f"Existing detail split ID does not match its partition: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmu-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    import pandas as pd

    lmu_root = args.lmu_data_root.expanduser().resolve()
    target = args.output_dir.expanduser().resolve()
    sources, source_provenance = validate_sources(lmu_root)
    protocol = {
        "heldout_fraction": HELDOUT_FRACTION,
        "paper_scope": "Tables II and III; all compared models use the same held-out rows",
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "source_manifest_sha256": source_provenance["manifest_sha256"],
    }
    existing = validate_existing(target, protocol, sources)
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    frames = {name: pd.read_csv(path, sep="\t") for name, path in sources.items()}
    mmbench_heldout, mmbench_split = split_mmbench(frames["MMBench_DEV_EN_V11"], args.seed)
    seed_heldout, seed_split = split_seed(frames["SEEDBench_IMG"], args.seed)
    split_results = {
        "MMBench_DEV_EN_V11": (mmbench_heldout, mmbench_split),
        "SEEDBench_IMG": (seed_heldout, seed_split),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.building-", dir=target.parent) as temporary:
        staging = Path(temporary)
        heldout_root = staging / "heldout"
        heldout_root.mkdir()
        datasets = {}
        partition_identity = {}
        for name, (heldout, split_info) in split_results.items():
            heldout_path = heldout_root / HELDOUT_FILENAMES[name]
            heldout.to_csv(heldout_path, sep="\t", index=False)
            row_count = len(frames[name])
            heldout_count_value = len(heldout)
            split_info.update(
                {
                    "heldout_fraction_actual": heldout_count_value / row_count,
                    "heldout_rows": heldout_count_value,
                    "source_rows": row_count,
                    "train_rows": row_count - heldout_count_value,
                }
            )
            relative = Path("heldout") / heldout_path.name
            datasets[name] = {
                **split_info,
                "heldout_tsv": str((target / relative).resolve()),
                "heldout_tsv_bytes": heldout_path.stat().st_size,
                "heldout_tsv_sha256": sha256_file(heldout_path),
                "source": source_provenance["datasets"][name],
            }
            partition_identity[name] = {
                "heldout_indices_sha256": split_info["heldout_indices_sha256"],
                "train_indices_sha256": split_info["train_indices_sha256"],
            }
        split_id = sha256_bytes(
            json.dumps(
                {"partition": partition_identity, "protocol": protocol},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": datasets,
            "generator_script_sha256": sha256_file(Path(__file__)),
            "protocol": protocol,
            "source_provenance": source_provenance,
            "split_id": split_id,
        }
        (staging / "detail_split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, target)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
