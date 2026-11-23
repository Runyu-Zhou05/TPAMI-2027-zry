"""CPU-only contracts for Golden validation result summaries."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/golden/summarize_golden_validation.py"


class GoldenValidationSummaryTest(unittest.TestCase):
    def write_step(self, validation_dir: Path, step: int, scores: list[float], row_step: int | None = None) -> None:
        records = [
            {"input": f"question-{index}", "output": "A", "score": score, "step": step if row_step is None else row_step}
            for index, score in enumerate(scores)
        ]
        (validation_dir / f"{step}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def run_summary(
        self, validation_dir: Path, output_dir: Path, expected_steps: tuple[int, ...] = (0, 20, 40)
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--validation-data-dir",
                str(validation_dir),
                "--output-dir",
                str(output_dir),
                "--expected-steps",
                ",".join(str(step) for step in expected_steps),
            ],
            capture_output=True,
            text=True,
        )

    def test_writes_csv_and_json_with_expected_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_dir = root / "validation"
            validation_dir.mkdir()
            self.write_step(validation_dir, 0, [0, 1, 1, 0])
            self.write_step(validation_dir, 20, [1, 1, 1, 0])
            self.write_step(validation_dir, 40, [1, 1, 1, 1])
            output_dir = root / "summary"

            completed = self.run_summary(validation_dir, output_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((output_dir / "golden_validation_summary.json").read_text())
            self.assertEqual(
                payload,
                [
                    {"step": 0, "accuracy": 0.5, "correct": 2, "rows": 4},
                    {"step": 20, "accuracy": 0.75, "correct": 3, "rows": 4},
                    {"step": 40, "accuracy": 1.0, "correct": 4, "rows": 4},
                ],
            )
            with (output_dir / "golden_validation_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0], {"step": "0", "accuracy": "0.5", "correct": "2", "rows": "4"})
            self.assertEqual(list(rows[0]), ["step", "accuracy", "correct", "rows"])

    def test_rejects_row_count_mismatch_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_dir = root / "validation"
            validation_dir.mkdir()
            self.write_step(validation_dir, 0, [0, 1])
            self.write_step(validation_dir, 20, [0])
            self.write_step(validation_dir, 40, [1, 1])
            output_dir = root / "summary"

            completed = self.run_summary(validation_dir, output_dir)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("row-count mismatch", completed.stderr)
            self.assertFalse(output_dir.exists())

    def test_rejects_nonbinary_nonfinite_and_wrong_step_scores(self):
        invalid_rows = (
            (0.5, 0, "finite 0 or 1"),
            (float("nan"), 0, "finite 0 or 1"),
            (1, 20, "Step field mismatch"),
        )
        for score, row_step, message in invalid_rows:
            with self.subTest(score=score, row_step=row_step), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                validation_dir = root / "validation"
                validation_dir.mkdir()
                self.write_step(validation_dir, 0, [score], row_step=row_step)
                self.write_step(validation_dir, 20, [1])
                self.write_step(validation_dir, 40, [1])

                completed = self.run_summary(validation_dir, root / "summary")
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)

    def test_refuses_to_overwrite_an_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_dir = root / "validation"
            validation_dir.mkdir()
            for step in (0, 20, 40):
                self.write_step(validation_dir, step, [1])
            output_dir = root / "summary"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("unchanged\n")

            completed = self.run_summary(validation_dir, output_dir)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Refusing to reuse", completed.stderr)
            self.assertEqual(sentinel.read_text(), "unchanged\n")


if __name__ == "__main__":
    unittest.main()
