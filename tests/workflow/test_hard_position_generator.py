"""CPU-only unit tests for the MMBench PositionQA grid helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_BENCHMARKS = ROOT / "build_benchmarks"
sys.path.insert(0, str(BUILD_BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "hard_position_generator", BUILD_BENCHMARKS / "makeposition_mmbench.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HardPositionGeneratorTest(unittest.TestCase):
    def test_default_grid_preserves_original_contract(self):
        args = MODULE.parse_args([])
        self.assertEqual(args.grid_size, 2)
        self.assertEqual(
            MODULE.get_grid_labels(2),
            ("top-left", "top-right", "bottom-left", "bottom-right"),
        )
        self.assertEqual(MODULE.get_grid_bounds(7, 9, 2)["top-left"], (0, 3, 0, 4))
        self.assertEqual(MODULE.get_grid_bounds(7, 9, 2)["bottom-right"], (3, 7, 4, 9))
        self.assertIs(MODULE.get_query_template(2), MODULE.QUERY_TEMPLATE)
        self.assertEqual(
            hashlib.sha256(MODULE.get_query_template(2).encode()).hexdigest(),
            "379d25e13c93ade8108151f90740c573a95038091555cb752edafa7b303c96f6",
        )
        self.assertEqual(MODULE.get_output_directory(2), "datasets/MMBench-PositionQA")

    def test_three_by_three_labels_bounds_prompt_and_output(self):
        labels = MODULE.get_grid_labels(3)
        self.assertEqual(
            labels,
            (
                "top-left",
                "top-center",
                "top-right",
                "middle-left",
                "middle-center",
                "middle-right",
                "bottom-left",
                "bottom-center",
                "bottom-right",
            ),
        )

        bounds = MODULE.get_grid_bounds(8, 10, 3)
        self.assertEqual(bounds["top-left"], (0, 2, 0, 3))
        self.assertEqual(bounds["middle-center"], (2, 5, 3, 6))
        self.assertEqual(bounds["bottom-right"], (5, 8, 6, 10))

        coverage = [[0] * 10 for _ in range(8)]
        for top, bottom, left, right in bounds.values():
            for row in range(top, bottom):
                for column in range(left, right):
                    coverage[row][column] += 1
        self.assertTrue(all(count == 1 for row in coverage for count in row))

        prompt = MODULE.get_query_template(3)
        self.assertIn("nine cases", prompt)
        for label in labels:
            self.assertIn(label, prompt)
        self.assertEqual(MODULE.get_output_directory(3), "datasets/MMBench-PositionQA_3x3")

    def test_cli_accepts_only_supported_grid_sizes(self):
        self.assertEqual(MODULE.parse_args(["--grid-size", "3"]).grid_size, 3)
        with self.assertRaises(SystemExit):
            MODULE.parse_args(["--grid-size", "4"])


if __name__ == "__main__":
    unittest.main()
