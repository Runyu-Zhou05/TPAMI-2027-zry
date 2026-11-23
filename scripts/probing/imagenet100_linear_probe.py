#!/usr/bin/env python3
"""Frozen pre/post-merger ImageNet-100 linear probes for Qwen2.5-VL."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROLES = ("base", "rotation", "position")
FEATURE_LEVELS = ("pre_merger", "post_merger")
EXPECTED_SPLITS = {"train": (100, 130_000, 1_300), "val": (100, 5_000, 50)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def dataset_inventory(data_root: Path) -> dict:
    data_root = data_root.expanduser().resolve()
    records = {}
    class_sets = {}
    for split, (expected_classes, expected_images, expected_per_class) in EXPECTED_SPLITS.items():
        split_root = data_root / split
        if not split_root.is_dir():
            raise SystemExit(f"ImageNet-100 split is missing: {split_root}")
        class_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
        counts = {
            path.name: sum(
                1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
            for path in class_dirs
        }
        if (
            len(counts) != expected_classes
            or sum(counts.values()) != expected_images
            or set(counts.values()) != {expected_per_class}
        ):
            raise SystemExit(
                f"Unexpected ImageNet-100 {split} inventory: classes={len(counts)}, "
                f"images={sum(counts.values())}, per_class={Counter(counts.values())}"
            )
        class_sets[split] = set(counts)
        records[split] = {
            "class_count": len(counts),
            "images": sum(counts.values()),
            "images_per_class": expected_per_class,
        }
    if class_sets["train"] != class_sets["val"]:
        raise SystemExit("ImageNet-100 train/val class identities differ")
    classes = sorted(class_sets["train"])
    class_list_sha256 = hashlib.sha256(("\n".join(classes) + "\n").encode()).hexdigest()
    labels_path = data_root / "Labels.json"
    if not labels_path.is_file():
        raise SystemExit(f"ImageNet-100 label metadata is missing: {labels_path}")
    labels = json.loads(labels_path.read_text())
    if not isinstance(labels, dict) or set(labels) != set(classes):
        raise SystemExit("ImageNet-100 Labels.json does not match class directories")
    return {
        "class_list": classes,
        "class_list_sha256": class_list_sha256,
        "labels_json": str(labels_path),
        "labels_json_sha256": sha256_file(labels_path),
        "root": str(data_root),
        "splits": records,
    }


def model_identity(model_path: Path, model_role: str) -> dict:
    model_path = model_path.expanduser().resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Model config is missing: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "qwen2_5_vl":
        raise SystemExit(f"Expected qwen2_5_vl model, got {config.get('model_type')!r}")
    export_path = model_path / "ssl4rl_export_manifest.json"
    export_record = None
    if model_role == "base":
        if export_path.is_file():
            raise SystemExit("Base probe must not use a trained SSL4RL export")
    else:
        if not export_path.is_file():
            raise SystemExit(f"Normal {model_role} probe requires {export_path}")
        export = json.loads(export_path.read_text())
        if export.get("artifact_type") != "ssl4rl_export" or export.get("model_role") != model_role:
            raise SystemExit(f"SSL4RL export identity mismatch: {export_path}")
        export_record = {"path": str(export_path), "sha256": sha256_file(export_path)}
    return {
        "config_sha256": sha256_file(config_path),
        "export_manifest": export_record,
        "model_path": str(model_path),
        "model_role": model_role,
    }


def selected_indices(samples: list[tuple[str, int]], class_count: int, per_class: int, seed: int) -> tuple[list[int], dict[int, int]]:
    grouped = defaultdict(list)
    for index, (_, label) in enumerate(samples):
        grouped[int(label)].append(index)
    labels = sorted(grouped)
    if class_count < 1 or class_count > len(labels):
        raise ValueError(f"class_count must be in [1,{len(labels)}]")
    rng = random.Random(seed)
    chosen_labels = sorted(rng.sample(labels, class_count)) if class_count < len(labels) else labels
    remap = {label: new_label for new_label, label in enumerate(chosen_labels)}
    indices = []
    for label in chosen_labels:
        candidates = list(grouped[label])
        if per_class < 1 or per_class > len(candidates):
            raise ValueError(f"per_class={per_class} is invalid for label {label} with {len(candidates)} samples")
        indices.extend(rng.sample(candidates, per_class))
    indices.sort()
    return indices, remap


def load_probe_runtime(model_path: Path, device: str, image_size: int, attn_implementation: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    pixels = image_size * image_size
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=pixels,
        max_pixels=pixels,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    model.requires_grad_(False)
    visual = model.visual
    del model
    gc.collect()
    visual.to(device)
    visual.eval()
    if any(parameter.requires_grad for parameter in visual.parameters()):
        raise RuntimeError("Visual feature extractor is not fully frozen")
    return processor, visual


def pool_visual_features(visual, pixel_values, grid_thw):
    import torch

    captured = {}

    def capture_pre_merger(_module, inputs):
        captured["pre_merger"] = inputs[0].detach()

    hook = visual.merger.register_forward_pre_hook(capture_pre_merger)
    try:
        with torch.inference_mode():
            post_merger = visual(pixel_values, grid_thw=grid_thw)
    finally:
        hook.remove()
    pre_merger = captured.get("pre_merger")
    if pre_merger is None:
        raise RuntimeError("Visual merger pre-hook did not capture features")
    grids = grid_thw.detach().cpu().tolist()
    merge = int(visual.spatial_merge_size)
    pre_counts = [int(t * h * w) for t, h, w in grids]
    post_counts = [count // (merge * merge) for count in pre_counts]
    if sum(pre_counts) != pre_merger.shape[0] or sum(post_counts) != post_merger.shape[0]:
        raise RuntimeError(
            f"Visual token accounting mismatch: pre={pre_merger.shape[0]}/{sum(pre_counts)}, "
            f"post={post_merger.shape[0]}/{sum(post_counts)}"
        )
    pooled_pre = torch.stack([chunk.float().mean(0) for chunk in torch.split(pre_merger, pre_counts)])
    pooled_post = torch.stack([chunk.float().mean(0) for chunk in torch.split(post_merger, post_counts)])
    return {"pre_merger": pooled_pre, "post_merger": pooled_post}, {
        "pre_tokens_per_image": pre_counts,
        "post_tokens_per_image": post_counts,
    }


def extract_cache(
    *,
    model_path: Path,
    model_role: str,
    data_root: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    workers: int,
    image_size: int,
    attn_implementation: str,
    class_count: int | None,
    train_per_class: int | None,
    val_per_class: int | None,
    seed: int,
) -> Path:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import InterpolationMode, v2

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse feature-cache directory: {output_dir}")
    inventory = dataset_inventory(data_root)
    identity = model_identity(model_path, model_role)
    transform = v2.Compose(
        [
            v2.Resize(256, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(image_size),
        ]
    )

    class ProbeDataset(Dataset):
        def __init__(self, folder: ImageFolder, indices: list[int], remap: dict[int, int]):
            self.folder = folder
            self.indices = indices
            self.remap = remap

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, position):
            path, label = self.folder.samples[self.indices[position]]
            with Image.open(path) as image:
                image = transform(image.convert("RGB"))
            return image, self.remap[int(label)], self.indices[position]

    def collate(rows):
        images, labels, indices = zip(*rows, strict=True)
        return list(images), torch.tensor(labels, dtype=torch.long), torch.tensor(indices, dtype=torch.long)

    folders = {split: ImageFolder(data_root / split) for split in EXPECTED_SPLITS}
    if folders["train"].class_to_idx != folders["val"].class_to_idx:
        raise SystemExit("ImageFolder train/val class mappings differ")
    total_classes = len(folders["train"].classes)
    if class_count is None:
        class_count = total_classes
    selections = {}
    selected_class_names = None
    for split, folder in folders.items():
        per_class = train_per_class if split == "train" else val_per_class
        if per_class is None:
            per_class = EXPECTED_SPLITS[split][2]
        indices, remap = selected_indices(folder.samples, class_count, per_class, seed)
        names = [folder.classes[label] for label in sorted(remap)]
        if selected_class_names is None:
            selected_class_names = names
        elif names != selected_class_names:
            raise SystemExit("Selected train/val probe classes differ")
        selections[split] = (ProbeDataset(folder, indices, remap), indices)

    processor, visual = load_probe_runtime(model_path, device, image_size, attn_implementation)
    staging_parent = output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=staging_parent) as temporary:
        staging = Path(temporary)
        split_records = {}
        feature_dims = None
        token_counts = {"pre_merger": set(), "post_merger": set()}
        for split in ("train", "val"):
            dataset, source_indices = selections[split]
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                collate_fn=collate,
            )
            feature_files = {}
            memmaps = {}
            labels_map = np.lib.format.open_memmap(
                staging / f"{split}_labels.npy", mode="w+", dtype=np.int64, shape=(len(dataset),)
            )
            indices_map = np.lib.format.open_memmap(
                staging / f"{split}_indices.npy", mode="w+", dtype=np.int64, shape=(len(dataset),)
            )
            offset = 0
            for images, labels, indices in loader:
                encoded = processor.image_processor(images=images, return_tensors="pt")
                pixel_values = encoded["pixel_values"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                grid_thw = encoded["image_grid_thw"].to(device=device, non_blocking=True)
                pooled, tokens = pool_visual_features(visual, pixel_values, grid_thw)
                if feature_dims is None:
                    feature_dims = {name: int(value.shape[1]) for name, value in pooled.items()}
                for name in FEATURE_LEVELS:
                    if name not in memmaps:
                        path = staging / f"{split}_{name}.npy"
                        memmaps[name] = np.lib.format.open_memmap(
                            path,
                            mode="w+",
                            dtype=np.float16,
                            shape=(len(dataset), feature_dims[name]),
                        )
                        feature_files[name] = path
                    array = pooled[name].cpu().numpy().astype(np.float16, copy=False)
                    memmaps[name][offset : offset + len(array)] = array
                    token_counts[name].update(tokens[f"{name.split('_')[0]}_tokens_per_image"])
                labels_map[offset : offset + len(labels)] = labels.numpy()
                indices_map[offset : offset + len(indices)] = indices.numpy()
                offset += len(labels)
            if offset != len(dataset):
                raise RuntimeError(f"Incomplete {split} extraction: {offset}/{len(dataset)}")
            for array in (*memmaps.values(), labels_map, indices_map):
                array.flush()
            split_records[split] = {
                "examples": len(dataset),
                "files": {
                    name: {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for name, path in {
                        **feature_files,
                        "labels": staging / f"{split}_labels.npy",
                        "indices": staging / f"{split}_indices.npy",
                    }.items()
                },
            }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": inventory,
            "extraction": {
                "attention_implementation": attn_implementation,
                "batch_size": batch_size,
                "class_count": class_count,
                "feature_dims": feature_dims,
                "feature_levels": list(FEATURE_LEVELS),
                "image_transform": f"Resize(256,BICUBIC)+CenterCrop({image_size})",
                "pooling": "mean over all valid spatial visual tokens",
                "seed": seed,
                "selected_classes": selected_class_names,
                "token_counts_observed": {name: sorted(values) for name, values in token_counts.items()},
                "train_per_class": len(selections["train"][1]) // class_count,
                "val_per_class": len(selections["val"][1]) // class_count,
            },
            "generator_git_commit": git_commit(),
            "model": identity,
            "schema_version": 1,
            "splits": split_records,
        }
        atomic_json(staging / "feature_cache_manifest.json", manifest)
        os.replace(staging, output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return output_dir / "feature_cache_manifest.json"


def fit_linear_heads(
    *,
    cache_dir: Path,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seeds: list[int],
) -> Path:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    cache_dir = cache_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse probe-result directory: {output_dir}")
    manifest_path = cache_dir / "feature_cache_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Feature-cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise SystemExit(f"Unsupported feature-cache manifest: {manifest_path}")
    for split in ("train", "val"):
        records = manifest.get("splits", {}).get(split, {}).get("files", {})
        if set(records) != {*FEATURE_LEVELS, "labels", "indices"}:
            raise SystemExit(f"Incomplete feature-cache inventory for {split}")
        for record in records.values():
            path = (cache_dir / record["path"]).resolve()
            if cache_dir not in path.parents or not path.is_file():
                raise SystemExit(f"Feature-cache artifact is missing or outside cache: {path}")
            if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
                raise SystemExit(f"Feature-cache artifact changed: {path}")

    class FeatureDataset(Dataset):
        def __init__(self, features: np.ndarray, labels: np.ndarray):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return np.array(self.features[index], dtype=np.float32, copy=True), int(self.labels[index])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        results = {}
        class_count = int(manifest["extraction"]["class_count"])
        for level in FEATURE_LEVELS:
            arrays = {}
            labels = {}
            for split in ("train", "val"):
                arrays[split] = np.load(cache_dir / f"{split}_{level}.npy", mmap_mode="r")
                labels[split] = np.load(cache_dir / f"{split}_labels.npy", mmap_mode="r")
                if len(arrays[split]) != len(labels[split]):
                    raise SystemExit(f"Feature/label row mismatch for {split}/{level}")
            seed_records = []
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                train_loader = DataLoader(
                    FeatureDataset(arrays["train"], labels["train"]),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    generator=torch.Generator().manual_seed(seed),
                )
                val_loader = DataLoader(
                    FeatureDataset(arrays["val"], labels["val"]),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
                head = nn.Linear(arrays["train"].shape[1], class_count).to(device)
                optimizer = torch.optim.SGD(
                    head.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                first_loss = None
                last_loss = None
                for _epoch in range(epochs):
                    head.train()
                    loss_sum = 0.0
                    examples = 0
                    for features, target in train_loader:
                        features = F.normalize(features.to(device, non_blocking=True), dim=-1)
                        target = target.to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(head(features), target)
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss.detach()) * len(target)
                        examples += len(target)
                    last_loss = loss_sum / examples
                    if first_loss is None:
                        first_loss = last_loss
                    scheduler.step()
                head.eval()
                correct = 0
                total = 0
                with torch.inference_mode():
                    for features, target in val_loader:
                        features = F.normalize(features.to(device, non_blocking=True), dim=-1)
                        target = target.to(device, non_blocking=True)
                        prediction = head(features).argmax(-1)
                        correct += int((prediction == target).sum())
                        total += len(target)
                record = {
                    "first_epoch_train_loss": first_loss,
                    "last_epoch_train_loss": last_loss,
                    "seed": seed,
                    "top1": correct / total,
                    "val_correct": correct,
                    "val_examples": total,
                }
                seed_records.append(record)
                torch.save(head.state_dict(), staging / f"{level}_seed{seed}_head.pt")
            top1 = [record["top1"] for record in seed_records]
            results[level] = {
                "mean_top1": float(np.mean(top1)),
                "seeds": seed_records,
                "std_top1": float(np.std(top1)),
            }
        result = {
            "cache_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "config": {
                "batch_size": batch_size,
                "epochs": epochs,
                "feature_normalization": "L2",
                "learning_rate": learning_rate,
                "optimizer": "SGD(momentum=0.9,weight_decay=0.0)+cosine",
                "seeds": seeds,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": manifest["model"],
            "results": results,
            "schema_version": 1,
        }
        atomic_json(staging / "linear_probe_results.json", result)
        os.replace(staging, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return output_dir / "linear_probe_results.json"


def add_extract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-role", choices=MODEL_ROLES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    add_extract_arguments(extract_parser)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--cache-dir", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--device", default="cuda:0")
    fit_parser.add_argument("--epochs", type=int, default=90)
    fit_parser.add_argument("--batch-size", type=int, default=4096)
    fit_parser.add_argument("--learning-rate", type=float, default=0.1)
    fit_parser.add_argument("--seeds", default="0,1,2")
    smoke_parser = subparsers.add_parser("smoke")
    add_extract_arguments(smoke_parser)
    smoke_parser.add_argument("--classes", type=int, default=4)
    smoke_parser.add_argument("--train-per-class", type=int, default=8)
    smoke_parser.add_argument("--val-per-class", type=int, default=4)
    smoke_parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.command in {"extract", "smoke"}:
        if args.batch_size < 1 or args.workers < 0 or args.image_size < 28:
            raise SystemExit("Invalid extraction batch/workers/image-size configuration")
    if args.command == "extract":
        extract_cache(
            model_path=args.model_path,
            model_role=args.model_role,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
            attn_implementation=args.attn_implementation,
            class_count=None,
            train_per_class=None,
            val_per_class=None,
            seed=args.seed,
        )
    elif args.command == "fit":
        seeds = [int(value) for value in args.seeds.split(",") if value]
        if not seeds or args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
            raise SystemExit("Invalid linear-probe fitting configuration")
        fit_linear_heads(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seeds=seeds,
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
        cache_dir = output_dir / "features"
        results_dir = output_dir / "results"
        if output_dir.exists():
            raise SystemExit(f"Refusing to reuse smoke directory: {output_dir}")
        output_dir.mkdir(parents=True)
        extract_cache(
            model_path=args.model_path,
            model_role=args.model_role,
            data_root=args.data_root,
            output_dir=cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
            attn_implementation=args.attn_implementation,
            class_count=args.classes,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            seed=args.seed,
        )
        result_path = fit_linear_heads(
            cache_dir=cache_dir,
            output_dir=results_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=min(128, args.classes * args.train_per_class),
            learning_rate=0.1,
            seeds=[0],
        )
        result = json.loads(result_path.read_text())
        for level, record in result["results"].items():
            seed_record = record["seeds"][0]
            if not math.isfinite(seed_record["last_epoch_train_loss"]):
                raise SystemExit(f"Non-finite smoke loss for {level}")
            if seed_record["last_epoch_train_loss"] >= seed_record["first_epoch_train_loss"]:
                raise SystemExit(f"Smoke linear head did not reduce loss for {level}")
        atomic_json(
            output_dir / "smoke_manifest.json",
            {
                "feature_cache_manifest": str(cache_dir / "feature_cache_manifest.json"),
                "linear_probe_results": str(result_path),
                "model_loaded": True,
                "status": "PASS",
            },
        )
        print(json.dumps({"status": "PASS", "smoke_manifest": str(output_dir / "smoke_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
