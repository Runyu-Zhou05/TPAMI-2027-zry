"""CPU-only contracts for MathVista/MMMU reasoning evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts" / "evaluation"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run(root: Path, role: str, judge: str = "gpt-4o-mini") -> Path:
    attempt = root / role
    attempt.mkdir()
    predictions = {}
    scores = {}
    for benchmark in ("MathVista_MINI", "MMMU_DEV_VAL"):
        prediction = attempt / f"{benchmark}.xlsx"
        score = attempt / f"{benchmark}.csv"
        prediction.write_bytes(f"prediction-{role}-{benchmark}".encode())
        score.write_text(f"score-{role}-{benchmark}\n")
        predictions[benchmark] = {
            "path": str(prediction.resolve()),
            "rows": 1,
            "sha256": sha256(prediction),
            "sorted_index_sha256": "a" * 64,
        }
        scores[benchmark] = {"path": str(score.resolve()), "sha256": sha256(score)}
    model_config = {
        "class": "Qwen2VLChat",
        "max_new_tokens": 4096,
        "model_path": f"/{role}",
        "temperature": 0.01,
    }
    configs = {
        name: {"data": {name: {"dataset": name}}, "model": {role: model_config}}
        for name in ("MathVista_MINI", "MMMU_DEV_VAL")
    }
    provenance = {
        "configs": configs,
        "data": {"same": True},
        "data_manifest_sha256": "data",
        "evaluator": {"same": True},
        "judges": {"MathVista_MINI": judge, "MMMU_DEV_VAL": "exact_matching"},
        "model": {"role": role},
        "model_role": role,
        "package_freeze": ["fixture==1"],
        "reasoning_format": "<think>complete reasoning</think> followed by <answer>final answer</answer>",
    }
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "attempt_fingerprint": fingerprint,
        "benchmarks": ["MathVista_MINI", "MMMU_DEV_VAL"],
        "evaluation_role": "reasoning_two_benchmarks",
        "model_role": role,
        "prediction_artifacts": predictions,
        "provenance": provenance,
        "reasoning_coverage": {
            name: {"formatted_reasoning_rate": 1.0, "formatted_reasoning_rows": 1, "nonempty_rows": 1, "rows": 1}
            for name in predictions
        },
        "run_id": "run-1",
        "schema_version": 1,
        "score_artifacts": scores,
        "score_input_prediction_sha256": {
            name: record["sha256"] for name, record in predictions.items()
        },
        "scores_percent": {"MathVista_MINI": 50.0, "MMMU_DEV_VAL": 60.0},
    }
    (attempt / "attempt_preflight.json").write_text(
        json.dumps({"fingerprint": fingerprint, "provenance": provenance})
    )
    for name, config in configs.items():
        (attempt / f"vlmeval_config_{name}.json").write_text(json.dumps(config))
    manifest_path = attempt / "reasoning_benchmark_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


class ReasoningWorkflowTest(unittest.TestCase):
    def test_deepseek_judge_patch_is_idempotent_and_redacts_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            judge = root / "judge_util.py"
            wrapper = root / "gpt.py"
            judge.write_text(
                "model_map = {\n"
                "            'deepseek': 'deepseek-ai/DeepSeek-V2.5',\n"
                "}\n"
            )
            wrapper.write_text(
                "class Wrapper:\n"
                "    def f(self):\n"
                "        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')\n"
            )
            command = [
                sys.executable,
                str(EVAL / "patch_vlmeval_deepseek_judge.py"),
                str(judge),
                str(wrapper),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = (judge.read_text(), wrapper.read_text())
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual((judge.read_text(), wrapper.read_text()), first)
            self.assertIn("'deepseek-chat': 'deepseek-chat'", first[0])
            self.assertIn("API Key: ***REDACTED***", first[1])
            self.assertNotIn("API Key: {self.key}", first[1])

    def test_mathvista_reasoning_patch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "image_vqa.py"
            source.write_text(
                "class ImageBaseDataset:\n    pass\n\n"
                "class MathVista(ImageBaseDataset):\n"
                "    # It returns a DataFrame\n"
                "    @classmethod\n"
                "    def evaluate(self, eval_file, **judge_kwargs):\n"
                "        return None\n"
            )
            command = [sys.executable, str(EVAL / "patch_vlmeval_reasoning_prompts.py"), str(source)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = source.read_text()
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(source.read_text(), first)
            self.assertIn("<think> and </think>", first)
            self.assertIn("<answer>final answer</answer>", first)

    def test_rotation_run_summarizes_without_base_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rotation = build_run(root, "rotation")
            output = root / "results"
            result = subprocess.run(
                [
                    sys.executable,
                    str(EVAL / "aggregate_reasoning_results.py"),
                    "--rotation-manifest",
                    str(rotation),
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((output / "reasoning_results.json").read_text())
            self.assertEqual(payload["benchmarks"]["MMMU_DEV_VAL"]["rotation"], 60.0)
            self.assertNotIn("base_manifest", payload)

    def test_base_manifest_is_rejected_as_rotation_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = build_run(root, "base", judge="deepseek-chat")
            result = subprocess.run(
                [
                    sys.executable,
                    str(EVAL / "aggregate_reasoning_results.py"),
                    "--rotation-manifest",
                    str(base),
                    "--output-dir",
                    str(root / "results"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid rotation reasoning manifest identity", result.stderr)

    def test_protocol_constants_are_pinned(self):
        source = (EVAL / "prepare_reasoning_data.py").read_text()
        self.assertIn("f199b98e178e5a2a20e7048f5dcb0464", source)
        self.assertIn("585e8ad75e73f75dcad265dfd0417d64", source)
        runner = (EVAL / "run_reasoning_benchmarks.py").read_text()
        self.assertIn('parser.add_argument("--max-new-tokens", type=int, default=4096)', runner)
        self.assertIn('"MMMU_DEV_VAL": "exact_matching"', runner)
        self.assertIn('default="deepseek-chat"', runner)
        self.assertIn('choices=("rotation",)', runner)
        self.assertIn("https://api.deepseek.com/chat/completions", runner)
        full_runner = (EVAL / "run_six_benchmarks.py").read_text()
        self.assertIn('("paper-readme", "base", "mmbench")', full_runner)
        self.assertIn('("paper-readme", "rotation", "six")', full_runner)
        self.assertIn("min_pixels = 35_840", full_runner)
        detail_runner = (EVAL / "run_detail_benchmarks.py").read_text()
        self.assertIn('choices=("rotation",)', detail_runner)


if __name__ == "__main__":
    unittest.main()
