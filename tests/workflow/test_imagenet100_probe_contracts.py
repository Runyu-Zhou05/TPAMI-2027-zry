"""CPU-only contracts for the frozen ImageNet-100 visual-feature probes."""

from __future__ import annotations

import importlib
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBING = ROOT / "scripts" / "probing"
if str(PROBING) not in sys.path:
    sys.path.insert(0, str(PROBING))
probe = importlib.import_module("imagenet100_linear_probe")
aggregate = importlib.import_module("aggregate_imagenet100_probes")


class ImageNetProbeContractTest(unittest.TestCase):
    def test_balanced_selection_is_deterministic(self):
        samples = [(f"{label}-{index}.jpg", label) for label in range(3) for index in range(5)]
        first, first_remap = probe.selected_indices(samples, class_count=2, per_class=3, seed=42)
        second, second_remap = probe.selected_indices(samples, class_count=2, per_class=3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first_remap, second_remap)
        selected_labels = [samples[index][1] for index in first]
        self.assertEqual(sorted(selected_labels.count(label) for label in first_remap), [3, 3])

    def test_dataset_inventory_binds_classes_labels_and_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classes = ("n0001", "n0002")
            for split, per_class in (("train", 2), ("val", 1)):
                for name in classes:
                    directory = root / split / name
                    directory.mkdir(parents=True)
                    for index in range(per_class):
                        (directory / f"{index}.JPEG").write_bytes(b"fixture")
            (root / "Labels.json").write_text(json.dumps({name: name for name in classes}))
            original = probe.EXPECTED_SPLITS
            probe.EXPECTED_SPLITS = {"train": (2, 4, 2), "val": (2, 2, 1)}
            try:
                record = probe.dataset_inventory(root)
            finally:
                probe.EXPECTED_SPLITS = original
            self.assertEqual(record["splits"]["train"]["images"], 4)
            self.assertEqual(record["class_list"], list(classes))

    def test_model_identity_distinguishes_base_and_task_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
            self.assertEqual(probe.model_identity(model, "base")["model_role"], "base")
            (model / "ssl4rl_export_manifest.json").write_text(
                json.dumps({"artifact_type": "ssl4rl_export", "model_role": "position"})
            )
            self.assertEqual(probe.model_identity(model, "position")["model_role"], "position")
            with self.assertRaises(SystemExit):
                probe.model_identity(model, "rotation")
            for role in ("rotation", "jigsaw", "contrastive"):
                (model / "ssl4rl_export_manifest.json").write_text(
                    json.dumps({"artifact_type": "ssl4rl_export", "model_role": role})
                )
                self.assertEqual(probe.model_identity(model, role)["model_role"], role)

    def test_protocol_signature_ignores_model_identity_but_not_probe_settings(self):
        cache = {
            "dataset": {
                "class_list_sha256": "classes",
                "labels_json_sha256": "labels",
                "splits": {"train": {"images": 130000}, "val": {"images": 5000}},
            },
            "extraction": {
                "attention_implementation": "sdpa",
                "class_count": 100,
                "feature_dims": {"pre_merger": 1280, "post_merger": 2048},
                "feature_levels": ["pre_merger", "post_merger"],
                "image_transform": "Resize(256,BICUBIC)+CenterCrop(224)",
                "pooling": "mean over all valid spatial visual tokens",
                "seed": 42,
                "selected_classes": ["a", "b"],
                "train_per_class": 1300,
                "val_per_class": 50,
            },
        }
        result = {"config": {"epochs": 90}}
        first = aggregate.protocol_signature(result, cache)
        cache["extraction"]["seed"] = 43
        second = aggregate.protocol_signature(result, cache)
        self.assertNotEqual(first, second)

    def test_feature_extractor_uses_merger_hook_and_freezes_visual_stack(self):
        source = (PROBING / "imagenet100_linear_probe.py").read_text()
        self.assertIn("visual.merger.register_forward_pre_hook", source)
        self.assertIn("model.requires_grad_(False)", source)
        self.assertIn('FEATURE_LEVELS = ("pre_merger", "post_merger")', source)

    def test_base_and_four_compatible_full_results_are_aggregated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_paths = {}
            for offset, role in enumerate(("base", "rotation", "position", "jigsaw", "contrastive")):
                role_root = root / role
                role_root.mkdir()
                cache = {
                    "schema_version": 1,
                    "model": {"model_role": role},
                    "dataset": {
                        "class_list_sha256": "classes",
                        "labels_json_sha256": "labels",
                        "splits": {"train": {"images": 130000}, "val": {"images": 5000}},
                    },
                    "extraction": {
                        "attention_implementation": "sdpa",
                        "class_count": 100,
                        "feature_dims": {"pre_merger": 1280, "post_merger": 2048},
                        "feature_levels": ["pre_merger", "post_merger"],
                        "image_transform": "Resize(256,BICUBIC)+CenterCrop(224)",
                        "pooling": "mean over all valid spatial visual tokens",
                        "seed": 42,
                        "selected_classes": [f"class-{i}" for i in range(100)],
                        "train_per_class": 1300,
                        "val_per_class": 50,
                    },
                }
                cache_path = role_root / "feature_cache_manifest.json"
                cache_path.write_text(json.dumps(cache))
                result = {
                    "schema_version": 1,
                    "model": {"model_role": role},
                    "cache_manifest": {
                        "path": str(cache_path),
                        "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                    },
                    "config": {"epochs": 90, "seeds": [0, 1, 2]},
                    "results": {
                        level: {"mean_top1": 0.50 + offset * 0.01}
                        for level in ("pre_merger", "post_merger")
                    },
                }
                result_path = role_root / "linear_probe_results.json"
                result_path.write_text(json.dumps(result))
                result_paths[role] = result_path
            output = root / "summary"
            command = [sys.executable, str(PROBING / "aggregate_imagenet100_probes.py")]
            for role, path in result_paths.items():
                command.extend([f"--{role}-result", str(path)])
            command.extend(["--output-dir", str(output)])
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = (output / "imagenet100_probe_results.csv").read_text().splitlines()
            self.assertEqual(
                rows[0],
                "Feature,Base,Rotation,Position,Jigsaw,Contrastive,"
                "Rotation-Base,Position-Base,Jigsaw-Base,Contrastive-Base",
            )
            self.assertIn("+1.00,+2.00,+3.00,+4.00", rows[1])


if __name__ == "__main__":
    unittest.main()
