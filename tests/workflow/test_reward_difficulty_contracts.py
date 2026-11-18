"""Dependency-light tests for reviewer-facing GRPO reward diversity metrics."""

from __future__ import annotations

import math
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "reward_difficulty_contract_module", ROOT / "verl/utils/reward_difficulty.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
aggregate_group_statistics = MODULE.aggregate_group_statistics
group_reward_statistics = MODULE.group_reward_statistics


class RewardDifficultyMetricTest(unittest.TestCase):
    def test_binary_collapse_and_informative_groups(self):
        groups = [
            group_reward_statistics([0, 0, 0, 0, 0]),
            group_reward_statistics([1, 1, 1, 1, 1]),
            group_reward_statistics([0, 1, 0, 1, 0]),
        ]
        aggregate = aggregate_group_statistics(groups)
        self.assertAlmostEqual(aggregate["informative_group_fraction"], 1 / 3)
        self.assertAlmostEqual(aggregate["all_fail_group_fraction"], 1 / 3)
        self.assertAlmostEqual(aggregate["all_success_group_fraction"], 1 / 3)
        self.assertAlmostEqual(aggregate["mixed_success_group_fraction"], 1 / 3)
        self.assertAlmostEqual(groups[2]["reward_entropy_bits"], -(0.4 * math.log2(0.4) + 0.6 * math.log2(0.6)))
        self.assertAlmostEqual(groups[2]["reward_entropy_normalized"], groups[2]["reward_entropy_bits"])

    def test_global_variance_does_not_imply_grpo_signal(self):
        groups = [
            group_reward_statistics([0, 0, 0, 0, 0]),
            group_reward_statistics([1, 1, 1, 1, 1]),
        ]
        aggregate = aggregate_group_statistics(groups)
        self.assertEqual(aggregate["informative_group_fraction"], 0.0)
        self.assertEqual(aggregate["mean_group_reward_variance"], 0.0)
        self.assertAlmostEqual(aggregate["global_reward_std"], 0.5)

    def test_continuous_entropy_is_empirical_after_rounding(self):
        record = group_reward_statistics([0.1, 0.1, 0.2, 0.2, 0.2])
        self.assertTrue(record["informative"])
        self.assertEqual(record["unique_reward_count"], 2)
        self.assertGreater(record["reward_entropy_bits"], 0)

    def test_group_requires_multiple_finite_rollouts(self):
        with self.assertRaises(ValueError):
            group_reward_statistics([1.0])
        with self.assertRaises(ValueError):
            group_reward_statistics([0.0, float("nan")])


if __name__ == "__main__":
    unittest.main()
