"""Dependency-light contracts for Table-XI hard-variant preprocessing."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/difficulty/prepare_hard_variant_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hard_variant_data_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HardVariantDataContractTest(unittest.TestCase):
    def test_task_registry_matches_generators_and_image_counts(self):
        module = load_module()
        self.assertEqual(
            module.TASKS,
            {
                "position3x3": {"dataset": "MMBench-PositionQA_3x3", "images": 2},
                "rotation45": {"dataset": "MMBench_RotationQA_45degree", "images": 2},
                "jigsaw3x3": {"dataset": "MMBench_JigsawQA_3x3", "images": 9},
                "contrastive_strong": {"dataset": "MMBench-ContrastiveQA_strong", "images": 2},
            },
        )

    def test_cli_exposes_all_tasks_without_heavy_dependencies(self):
        completed = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for task in ("position3x3", "rotation45", "jigsaw3x3", "contrastive_strong"):
            self.assertIn(task, completed.stdout)

    def test_preprocess_preserves_paper_split_mapping_and_reward_identity(self):
        source = SCRIPT.read_text()
        self.assertIn('"train": source["test"].map(', source)
        self.assertIn('"test": source["validation"].map(', source)
        self.assertIn('"reward_model": {"style": "rule", "ground_truth": answer}', source)
        self.assertIn('data_source = f"datasets/{spec[\'dataset\']}"', source)
        self.assertIn('record[0]["content"].count("<image>") != expected_images', source)


if __name__ == "__main__":
    unittest.main()
