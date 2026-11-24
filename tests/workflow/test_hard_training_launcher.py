from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "difficulty" / "run_mmbench_hard.py"
TASKS = {
    "position3x3": ("MMBench-PositionQA_3x3", 2),
    "rotation45": ("MMBench_RotationQA_45degree", 2),
    "jigsaw3x3": ("MMBench_JigsawQA_3x3", 9),
    "contrastive_strong": ("MMBench-ContrastiveQA_strong", 2),
}


class HardTrainingLauncherTests(unittest.TestCase):
    def dry_run(self, root: Path, task: str, mode: str = "train", *extra: str) -> str:
        command = [
            sys.executable,
            "-S",
            str(LAUNCHER),
            "--task",
            task,
            "--mode",
            mode,
            "--run-id",
            "contract-test",
            "--model-path",
            str(root / "base"),
            "--data-root",
            str(root / "data"),
            "--output-root",
            str(root / "output"),
            "--dry-run",
            *extra,
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_train_contract_for_all_tasks_and_dry_run_is_pure(self) -> None:
        for task, (dataset, image_limit) in TASKS.items():
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = self.dry_run(root, task)
                self.assertFalse((root / "output").exists())
                self.assertFalse((root / "base").exists())
                self.assertIn(f"{dataset}/train.parquet", output)
                self.assertIn(f"{dataset}/test.parquet", output)
                train_arg = next(line for line in output.splitlines() if "verl.trainer.main_ppo" in line)
                self.assertEqual(train_arg.count(f"{dataset}/test.parquet"), 2)
                self.assertIn("data.train_batch_size=512", output)
                self.assertIn("actor_rollout_ref.model.freeze_vision_encoder=False", output)
                self.assertIn("actor_rollout_ref.rollout.n=5", output)
                self.assertIn(f"+actor_rollout_ref.rollout.limit_images={image_limit}", output)
                self.assertIn("trainer.total_epochs=20", output)
                self.assertIn("trainer.total_training_steps=null", output)
                self.assertIn("trainer.save_freq=50", output)
                self.assertIn("trainer.test_freq=200", output)
                self.assertIn("trainer.resume_mode=disable", output)
                self.assertIn("trainer.reward_trace_dir=", output)
                self.assertIn(f"trainer.reward_trace_task={task}", output)

    def test_smoke_is_one_real_small_batch_in_an_independent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self.dry_run(root, "jigsaw3x3", "train")
            smoke = self.dry_run(root, "jigsaw3x3", "smoke")
            self.assertIn("ssl4rl_qwen_3b_mmbench_jigsaw3x3/checkpoints", train)
            self.assertIn("ssl4rl_qwen_3b_mmbench_jigsaw3x3_smoke/checkpoints", smoke)
            self.assertIn("data.train_batch_size=8", smoke)
            self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=8", smoke)
            self.assertIn("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1", smoke)
            self.assertIn("actor_rollout_ref.rollout.n=5", smoke)
            self.assertIn("trainer.total_epochs=1", smoke)
            self.assertIn("trainer.total_training_steps=1", smoke)
            self.assertIn("trainer.resume_mode=disable", smoke)
            self.assertIn("Global rollout batch=40", smoke)

    def test_reward_trace_can_be_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.dry_run(Path(temporary), "rotation45", "smoke", "--no-reward-trace")
            self.assertNotIn("trainer.reward_trace_dir=", output)
            self.assertNotIn("trainer.reward_trace_task=", output)
            self.assertNotIn("trainer.reward_trace_run_id=", output)

    def test_requires_exactly_eight_unique_numeric_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(LAUNCHER),
                    "--task",
                    "position3x3",
                    "--mode",
                    "train",
                    "--run-id",
                    "bad-gpus",
                    "--gpus",
                    "0,1,2,3,4,5,6,6",
                    "--output-root",
                    str(root / "output"),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly eight unique numeric GPU indices", result.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
