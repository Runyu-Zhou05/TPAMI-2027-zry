"""CPU-only tests for parameterized hard RotationQA and JigsawQA builders."""

from __future__ import annotations

import hashlib
import io
import random
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BUILDERS = ROOT / "build_benchmarks"
if str(BUILDERS) not in sys.path:
    sys.path.insert(0, str(BUILDERS))

import makejigsaw_mmbench as jigsaw  # noqa: E402
import makerotation_mmbench as rotation  # noqa: E402


class RotationGeneratorTest(unittest.TestCase):
    def test_cli_defaults_to_legacy_granularity_and_accepts_hard_mode(self):
        parser = rotation.build_parser()
        self.assertEqual(parser.parse_args([]).granularity, 90)
        self.assertEqual(parser.parse_args(["--granularity", "45"]).granularity, 45)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--granularity", "30"])

    def test_default_prompt_is_byte_compatible_and_hard_prompt_is_parameterized(self):
        self.assertFalse(hasattr(rotation, "QUERY_TEMPLATE"))
        legacy_prompt = rotation.build_query_template(90)
        self.assertEqual(
            hashlib.sha256(legacy_prompt.encode()).hexdigest(),
            "005d4a37d9223090eee904a116f84c407040b0ec39c8eeeb201f4dde68c8b9ba",
        )

        hard_prompt = rotation.build_query_template(45)
        self.assertEqual(hard_prompt.count("<image>"), 2)
        self.assertIn("granuality of rotation is 45 degrees", hard_prompt)
        self.assertIn("0, 45, 90, ..., 315 degrees", hard_prompt)
        self.assertEqual(rotation.possible_angles(45), tuple(range(0, 360, 45)))

    def test_seeded_hard_record_has_valid_answer_and_separate_directory(self):
        self.assertEqual(rotation.DEFAULT_SEED, 42)
        rng = random.Random(rotation.DEFAULT_SEED)
        angle = rng.choice(rotation.possible_angles(45))
        datum = {
            "index": 1,
            "source": "unit",
            "split": "test",
            "image": Image.new("RGB", (96, 96), "white"),
        }
        record = rotation.build_rotated_datum(datum, angle, granularity=45)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["answer"], str(angle))
        self.assertIn(int(record["answer"]), rotation.possible_angles(45))
        self.assertEqual(rotation.output_directory(45), "datasets/MMBench_RotationQA_45degree")
        self.assertEqual(rotation.output_directory(90), "datasets/MMBench_RotationQA_90degree")


class JigsawGeneratorTest(unittest.TestCase):
    def test_cli_defaults_to_legacy_grid_and_accepts_hard_mode(self):
        parser = jigsaw.build_parser()
        self.assertEqual(parser.parse_args([]).grid_size, 2)
        self.assertEqual(parser.parse_args(["--grid-size", "3"]).grid_size, 3)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--grid-size", "4"])

    def test_default_prompt_is_byte_compatible_and_hard_prompt_is_parameterized(self):
        self.assertFalse(hasattr(jigsaw, "QUERY_TEMPLATE"))
        legacy_prompt = jigsaw.build_query_template(2)
        self.assertEqual(
            hashlib.sha256(legacy_prompt.encode()).hexdigest(),
            "3e272927825f56923de91d52cd8c8b8e0be0f9d2d38c89af7e5f63fd72fb264d",
        )

        hard_prompt = jigsaw.build_query_template(3)
        self.assertEqual(hard_prompt.count("<image>"), 9)
        self.assertIn("9 parts of an original image", hard_prompt)
        self.assertIn("3x3 grid", hard_prompt)
        self.assertIn(jigsaw.ORDER_EXAMPLES[3], hard_prompt)

    def test_seeded_hard_record_has_inverse_answer_and_separate_directory(self):
        self.assertEqual(jigsaw.DEFAULT_SEED, 0)
        order = list(range(9))
        random.Random(jigsaw.DEFAULT_SEED).shuffle(order)
        datum = {
            "index": 2,
            "source": "unit",
            "split": "validation",
            "image": Image.new("RGB", (90, 90), "white"),
        }
        record = jigsaw.build_jigsaw_datum(datum, order, grid_size=3)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(order, [7, 5, 1, 3, 4, 2, 0, 8, 6])
        self.assertEqual(record["answer"], "7,3,6,4,5,2,9,1,8")
        self.assertEqual(len(record["images"]), 9)
        self.assertTrue(all(image.size == (30, 30) for image in record["images"]))
        self.assertEqual(jigsaw.output_directory(3), "datasets/MMBench_JigsawQA_3x3")
        self.assertEqual(jigsaw.output_directory(2), "datasets/MMBench_JigsawQA_2x2")


if __name__ == "__main__":
    unittest.main()
