#!/usr/bin/env python3
"""Run a bounded Qwen2.5-VL MMBench smoke through pinned VLMEvalKit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


VLMEVAL_COMMIT = "0a11917eede83742389b03d7f11fc252aabdf6ef"
DATASET = "MMBench_DEV_EN_V11"
MMBENCH_URL = "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv"
MMBENCH_MD5 = "30c05be8f2f347a50be25aa067248184"
L2_TO_TABLE = {
    "logic_reasoning": "Logical",
    "relation_reasoning": "Relation",
    "attribute_reasoning": "Attribute",
    "coarse_perception": "Coarse",
    "finegrained_perception (cross-instance)": "Cross-Inst.",
    "finegrained_perception (instance-level)": "Single-Inst.",
}
OFFICIAL_TO_TABLE = {
    "LR": "Logical",
    "RR": "Relation",
    "AR": "Attribute",
    "CP": "Coarse",
    "FP-C": "Cross-Inst.",
    "FP-S": "Single-Inst.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_tsv(source_tsv: Path | None, lmu_data_root: Path) -> Path:
    if source_tsv is not None:
        source_tsv = source_tsv.expanduser().resolve()
        if not source_tsv.is_file():
            raise SystemExit(f"Missing MMBench TSV: {source_tsv}")
        if md5_file(source_tsv) != MMBENCH_MD5:
            raise SystemExit(f"MMBench TSV checksum mismatch: {source_tsv}")
        return source_tsv

    target = lmu_data_root / f"{DATASET}.tsv"
    if target.is_file() and md5_file(target) == MMBENCH_MD5:
        return target
    temporary = target.with_suffix(".tsv.download")
    temporary.unlink(missing_ok=True)
    print(f"Downloading pinned {DATASET} from {MMBENCH_URL}", flush=True)
    urllib.request.urlretrieve(MMBENCH_URL, temporary)
    if md5_file(temporary) != MMBENCH_MD5:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded MMBench TSV failed MD5 validation: {MMBENCH_MD5}")
    os.replace(temporary, target)
    return target


def select_complete_circular_groups(data):
    import pandas as pd

    frame = data.copy()
    frame["_base_index"] = frame["index"].astype(int) % 1_000_000
    selected = []
    selection = {}
    for l2_category, table_name in L2_TO_TABLE.items():
        category_frame = frame[frame["l2-category"] == l2_category]
        if category_frame.empty:
            raise RuntimeError(f"MMBench data has no rows for {l2_category}")
        groups = [(int(base_index), group.copy()) for base_index, group in category_frame.groupby("_base_index")]
        groups.sort(key=lambda item: (-len(item[1]), item[0]))
        base_index, group = groups[0]
        if len(group) < 2:
            raise RuntimeError(f"No complete circular group found for {l2_category}")
        group = group.drop(columns=["_base_index"])
        selected.append(group)
        selection[table_name] = {
            "l2_category": l2_category,
            "base_index": base_index,
            "indices": [int(value) for value in sorted(group["index"].astype(int).tolist())],
        }
    return pd.concat(selected, ignore_index=True).sort_values("index").reset_index(drop=True), selection


def normalize_score_frame(score):
    import pandas as pd

    if isinstance(score, pd.DataFrame):
        return score
    if isinstance(score, dict):
        return pd.DataFrame([score])
    raise TypeError(f"Unexpected VLMEvalKit score type: {type(score).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-tsv", type=Path, default=None)
    parser.add_argument("--lmu-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    lmu_data_root = args.lmu_data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Invalid Qwen model directory: {model_path}")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    lmu_data_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_tsv = resolve_source_tsv(args.source_tsv, lmu_data_root)
    target_tsv = lmu_data_root / f"{DATASET}.tsv"
    if target_tsv.exists():
        if target_tsv.resolve() != source_tsv and sha256_file(target_tsv) != sha256_file(source_tsv):
            raise SystemExit(f"Existing {target_tsv} does not match {source_tsv}")
    else:
        target_tsv.symlink_to(source_tsv)

    os.environ["LMUData"] = str(lmu_data_root)
    import pandas as pd
    import torch
    import transformers
    import vlmeval
    from vlmeval.dataset import ImageMCQDataset
    from vlmeval.vlm import Qwen2VLChat

    vlmeval_root = Path(vlmeval.__file__).resolve().parent.parent
    marker = vlmeval_root / ".pinned-commit"
    installed_commit = marker.read_text().strip() if marker.is_file() else None
    if installed_commit != VLMEVAL_COMMIT:
        raise SystemExit(
            f"Expected VLMEvalKit {VLMEVAL_COMMIT}, found {installed_commit!r} at {vlmeval_root}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(
            f"Smoke requires exactly one visible CUDA GPU; found {torch.cuda.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES to one physical GPU."
        )

    dataset = ImageMCQDataset(dataset=DATASET)
    subset, selection = select_complete_circular_groups(dataset.data)
    subset_path = output_dir / "mmbench_v11_smoke_subset.tsv"
    subset.to_csv(subset_path, sep="\t", index=False)
    print(f"Selected {len(subset)} rows across all six MMBench subtasks: {selection}", flush=True)

    model = Qwen2VLChat(
        model_path=str(model_path),
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_new_tokens=args.max_new_tokens,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt=False,
        verbose=False,
    )
    predictions = []
    for position, (_, row) in enumerate(subset.iterrows(), start=1):
        message = dataset.build_prompt(row)
        prediction = model.generate(message, dataset=DATASET)
        predictions.append(prediction)
        print(
            f"[{position}/{len(subset)}] index={row['index']} "
            f"subtask={L2_TO_TABLE[row['l2-category']]} prediction={prediction!r}",
            flush=True,
        )

    prediction_frame = subset.copy()
    prediction_frame["prediction"] = predictions
    prediction_path = output_dir / "mmbench_v11_smoke_predictions.xlsx"
    prediction_frame.to_excel(prediction_path, index=False)
    score_frame = normalize_score_frame(dataset.evaluate(str(prediction_path), model="exact_matching", nproc=1))
    score_path = output_dir / "mmbench_v11_smoke_official_scores.csv"
    score_frame.to_csv(score_path, index=False)

    table2 = {}
    score_row = score_frame.iloc[0].to_dict()
    for official_name, table_name in OFFICIAL_TO_TABLE.items():
        value = score_row.get(official_name)
        table2[table_name] = None if pd.isna(value) else float(value) * 100.0
    overall = score_row.get("Overall")
    table2["Average"] = None if pd.isna(overall) else float(overall) * 100.0
    table2_path = output_dir / "mmbench_v11_smoke_table2.json"
    table2_path.write_text(json.dumps(table2, indent=2, sort_keys=False) + "\n")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset": DATASET,
        "gpu": torch.cuda.get_device_name(0),
        "max_new_tokens": args.max_new_tokens,
        "model_path": str(model_path),
        "prediction_file": str(prediction_path),
        "sample_count": len(subset),
        "selection": selection,
        "source_tsv": str(source_tsv),
        "source_tsv_sha256": sha256_file(source_tsv),
        "table2_smoke_percent": table2,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vlmeval_commit": installed_commit,
        "vlmeval_root": str(vlmeval_root),
    }
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
