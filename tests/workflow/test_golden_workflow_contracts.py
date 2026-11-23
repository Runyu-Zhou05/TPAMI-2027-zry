"""CPU-only contracts for the Position-to-Golden RLVR workflow."""

from __future__ import annotations

import base64
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_mmbench_golden_data_contract", GOLDEN_SCRIPTS / "prepare_mmbench_golden_data.py")
scorer = load_module("golden_vqa_contract", ROOT / "verl" / "utils" / "reward_score" / "golden_vqa.py")
if str(GOLDEN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GOLDEN_SCRIPTS))
if str(ROOT / "scripts" / "evaluation") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
launcher = load_module("run_mmbench_golden_contract", GOLDEN_SCRIPTS / "run_mmbench_golden.py")


def synthetic_old_mmbench() -> pd.DataFrame:
    """Build the pinned row shape with complete 2/3/4-row circular groups."""

    rows = []
    next_key = 1_000
    categories = sorted(prepare.EXPECTED_L2)
    for category_index, category in enumerate(categories):
        group_count = 241 if category_index < 3 else 240
        keys = [str(key) for key in range(next_key, next_key + group_count)]
        next_key += group_count
        ranked = sorted(keys, key=lambda key: (prepare.stable_rank(prepare.SEED, category, key), key))
        heldout_count = math.floor(group_count * prepare.HELDOUT_FRACTION + 0.5)
        two_row_key = ranked[0]
        four_row_key = ranked[heldout_count]
        for key in keys:
            size = 2 if key == two_row_key else 4 if key == four_row_key else 3
            rows.extend(
                {"index": int(key) + offset * 1_000_000, "l2-category": category}
                for offset in range(size)
            )
    return pd.DataFrame(rows)


class GoldenDataContractTest(unittest.TestCase):
    def test_variable_circular_groups_split_deterministically_60_40(self):
        frame = synthetic_old_mmbench()
        self.assertEqual(len(frame), 4_329)

        first_train, first_test, first_info = prepare.split_indices(frame, seed=42)
        second_train, second_test, second_info = prepare.split_indices(frame, seed=42)

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_test, second_test)
        self.assertEqual(first_info, second_info)
        self.assertEqual((len(first_train), len(first_test)), (2_607, 1_722))
        self.assertAlmostEqual(len(first_test) / len(frame), 0.4, places=2)
        self.assertEqual(first_info["group_size_histogram"], {"2": 6, "3": 1_431, "4": 6})

        train_keys = set(frame.loc[first_train, "index"].astype(int) % 1_000_000)
        test_keys = set(frame.loc[first_test, "index"].astype(int) % 1_000_000)
        self.assertTrue(train_keys.isdisjoint(test_keys))
        for category, counts in first_info["strata"].items():
            self.assertAlmostEqual(counts["heldout_groups"] / counts["total_groups"], 0.4, places=2, msg=category)

    def test_image_index_references_resolve_transitively(self):
        payload = b"\x89PNG\r\ncontract-image"
        encoded = base64.b64encode(payload).decode()
        frame = pd.DataFrame(
            [
                {"index": 17, "image": f"data:image/png;base64,{encoded}"},
                {"index": 1_000_017, "image": "17"},
                {"index": 2_000_017, "image": "1000017"},
            ]
        )
        resolved = prepare.resolve_image_bytes(frame)
        self.assertEqual(resolved, {17: payload, 1_000_017: payload, 2_000_017: payload})

    def test_cyclic_image_index_references_are_rejected(self):
        frame = pd.DataFrame([{"index": 1, "image": "2"}, {"index": 2, "image": "1"}])
        with self.assertRaisesRegex(SystemExit, "Cyclic MMBench image reference"):
            prepare.resolve_image_bytes(frame)


class GoldenRewardContractTest(unittest.TestCase):
    def test_strict_single_answer_tag_scoring(self):
        self.assertEqual(scorer.compute_score("reasoning\n<answer> B </answer>", "b"), 1.0)
        self.assertEqual(scorer.compute_score("reasoning\n<answer>B</answer>  \n", "B"), 1.0)
        self.assertEqual(scorer.compute_score("<answer>A</answer>", "B"), 0.0)
        for malformed in (
            "B",
            "<answer>AB</answer>",
            "<answer>B</answer><answer>B</answer>",
            "<answer>B",
            "<answer>B</answer> trailing garbage",
        ):
            self.assertEqual(scorer.compute_score(malformed, "B"), 0.0, malformed)
        with self.assertRaisesRegex(ValueError, "Invalid Golden VQA ground truth"):
            scorer.compute_score("<answer>A</answer>", "E")

    def test_reward_registry_routes_golden_data(self):
        source = (ROOT / "verl/utils/reward_score/__init__.py").read_text()
        self.assertIn('elif "MMBench_GoldenQA" in data_source:', source)
        self.assertIn("golden_vqa.compute_score", source)


class GoldenLauncherContractTest(unittest.TestCase):
    def test_dry_run_is_pure_and_pins_training_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "position-hf"
            data_root = root / "data"
            output_root = root / "output-must-not-exist"
            command = [
                sys.executable,
                "-S",
                str(GOLDEN_SCRIPTS / "run_mmbench_golden.py"),
                "--model-path",
                str(model),
                "--data-root",
                str(data_root),
                "--output-root",
                str(output_root),
                "--run-id",
                "golden-contract",
                "--gpus",
                "0,1,2,3,4,5,6,7",
                "--dry-run",
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(model.exists())
            self.assertFalse(data_root.exists())
            self.assertFalse(output_root.exists())
            self.assertIn("Expected budget: 20 epochs x 5 steps = 100", completed.stdout)
            self.assertIn("step 0 validation; steps 20,40,60,80,100 validation+checkpoint", completed.stdout)
            for override in (
                "trainer.total_epochs=20",
                "trainer.save_freq=20",
                "trainer.test_freq=20",
                "trainer.val_before_train=True",
                "trainer.resume_mode=disable",
                "trainer.total_training_steps=100",
                "data.filter_overlong_prompts=False",
                "data.truncation=error",
            ):
                self.assertIn(override, completed.stdout)

    def test_terminal_checkpoint_and_validation_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "checkpoints"
            validation = root / "validation"
            completion = checkpoints / "global_step_100" / "checkpoint_complete.json"
            completion.parent.mkdir(parents=True)
            validation.mkdir()
            (checkpoints / "latest_checkpointed_iteration.txt").write_text("100")
            completion.write_text(json.dumps({"schema_version": 2, "global_step": 100, "run_id": "run"}))
            for step in (0, 20, 40, 60, 80, 100):
                (validation / f"{step}.jsonl").write_text("{}\n")
            self.assertEqual(launcher.validate_completed_run(checkpoints, validation, "run")["global_step"], 100)
            (validation / "80.jsonl").unlink()
            with self.assertRaisesRegex(SystemExit, "missing steps"):
                launcher.validate_completed_run(checkpoints, validation, "run")


if __name__ == "__main__":
    unittest.main()
