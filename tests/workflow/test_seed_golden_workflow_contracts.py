"""CPU-only contracts for the SEED-Bench Position-to-Golden workflow."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SCRIPTS = ROOT / "scripts" / "golden"
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
EXPECTED_CATEGORY_COUNTS = {
    "Instance Attributes": 4_649,
    "Instance Identity": 1_831,
    "Instance Interaction": 97,
    "Instance Location": 978,
    "Instances Counting": 2_447,
    "Scene Understanding": 3_158,
    "Spatial Relation": 657,
    "Text Understanding": 84,
    "Visual Reasoning": 331,
}
EXPECTED_ROWS = {"train.parquet": 8_539, "test.parquet": 5_693}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "prepare_seedbench_golden_data_contract",
    GOLDEN_SCRIPTS / "prepare_seedbench_golden_data.py",
)
scorer = load_module("seed_golden_vqa_contract", ROOT / "verl/utils/reward_score/golden_vqa.py")
if str(GOLDEN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GOLDEN_SCRIPTS))
if str(ROOT / "scripts" / "evaluation") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
launcher = load_module("run_seedbench_golden_contract", GOLDEN_SCRIPTS / "run_seedbench_golden.py")


def synthetic_seedbench() -> pd.DataFrame:
    """Build the real row/category shape without carrying image payloads."""

    rows = []
    for category, count in EXPECTED_CATEGORY_COUNTS.items():
        rows.extend({"index": len(rows), "category": category} for _ in range(count))
    return pd.DataFrame(rows)


class SeedGoldenDataContractTest(unittest.TestCase):
    def test_real_source_and_output_constants(self):
        self.assertEqual(prepare.SOURCE_ROWS, 14_232)
        self.assertEqual(prepare.EXPECTED_CATEGORIES, EXPECTED_CATEGORIES)
        self.assertEqual(prepare.EXPECTED_OUTPUT_ROWS, EXPECTED_ROWS)

    def test_seed_42_split_is_deterministic_stratified_and_row_disjoint(self):
        frame = synthetic_seedbench()
        self.assertEqual(len(frame), 14_232)

        first_train, first_test, first_info = prepare.split_indices(frame, seed=42)
        second_train, second_test, second_info = prepare.split_indices(frame, seed=42)

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_test, second_test)
        self.assertEqual(first_info, second_info)
        self.assertEqual((len(first_train), len(first_test)), (8_539, 5_693))
        self.assertEqual(set(first_train) | set(first_test), set(frame.index))
        self.assertTrue(set(first_train).isdisjoint(first_test))
        self.assertEqual(first_info["seed"], 42)
        self.assertEqual(set(first_info["strata"]), EXPECTED_CATEGORIES)

        for category in EXPECTED_CATEGORIES:
            category_positions = set(frame.index[frame["category"] == category])
            heldout = len(category_positions & set(first_test))
            self.assertEqual(heldout, math.floor(len(category_positions) * 0.4 + 0.5))

    def test_duplicate_row_indices_are_rejected(self):
        frame = synthetic_seedbench()
        frame.loc[frame.index[-1], "index"] = frame.loc[frame.index[0], "index"]
        with self.assertRaisesRegex(SystemExit, "duplicate indices"):
            prepare.split_indices(frame, seed=42)


class SeedGoldenRewardContractTest(unittest.TestCase):
    def test_shared_reward_remains_strict(self):
        self.assertEqual(scorer.compute_score("reasoning\n<answer> C </answer>", "c"), 1.0)
        for malformed in ("C", "<answer>CC</answer>", "<answer>C</answer> trailing"):
            self.assertEqual(scorer.compute_score(malformed, "C"), 0.0, malformed)

    def test_reward_registry_routes_seed_golden_data(self):
        source = (ROOT / "verl/utils/reward_score/__init__.py").read_text()
        self.assertIn('"SEEDBench_GoldenQA" in data_source', source)
        self.assertIn("golden_vqa.compute_score", source)


class SeedGoldenLauncherContractTest(unittest.TestCase):
    def test_dry_run_is_pure_and_pins_position_initialized_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "position-hf"
            data_root = root / "data"
            output_root = root / "output-must-not-exist"
            command = [
                sys.executable,
                "-S",
                str(GOLDEN_SCRIPTS / "run_seedbench_golden.py"),
                "--model-path",
                str(model),
                "--data-root",
                str(data_root),
                "--output-root",
                str(output_root),
                "--run-id",
                "seed-golden-contract",
                "--gpus",
                "0,1,2,3,4,5,6,7",
                "--dry-run",
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(model.exists())
            self.assertFalse(data_root.exists())
            self.assertFalse(output_root.exists())
            self.assertIn("Expected budget: 20 epochs x 16 steps = 320", completed.stdout)
            self.assertIn(f"actor_rollout_ref.model.path={model.resolve()}", completed.stdout)
            for override in (
                "data.train_batch_size=512",
                "trainer.total_epochs=20",
                "trainer.save_freq=20",
                "trainer.test_freq=20",
                "trainer.val_before_train=True",
                "trainer.resume_mode=disable",
                "trainer.total_training_steps=320",
                "data.filter_overlong_prompts=False",
                "data.truncation=error",
            ):
                self.assertIn(override, completed.stdout)

    def test_terminal_checkpoint_and_all_validation_dumps_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "checkpoints"
            validation = root / "validation"
            completion = checkpoints / "global_step_320" / "checkpoint_complete.json"
            completion.parent.mkdir(parents=True)
            validation.mkdir()
            (checkpoints / "latest_checkpointed_iteration.txt").write_text("320")
            completion.write_text(json.dumps({"schema_version": 2, "global_step": 320, "run_id": "run"}))
            expected_steps = list(range(0, 321, 20))
            for step in expected_steps:
                (validation / f"{step}.jsonl").write_text("{}\n")

            result = launcher.validate_completed_run(checkpoints, validation, "run")
            self.assertEqual(result["global_step"], 320)
            self.assertEqual(result["validation_steps"], expected_steps)
            (validation / "300.jsonl").unlink()
            with self.assertRaisesRegex(SystemExit, "missing steps"):
                launcher.validate_completed_run(checkpoints, validation, "run")


if __name__ == "__main__":
    unittest.main()
