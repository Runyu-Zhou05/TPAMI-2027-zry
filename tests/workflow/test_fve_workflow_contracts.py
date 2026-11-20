"""CPU-only contract tests for the portable FVE workflow."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "scripts" / "evaluation"
FVE_SCRIPTS = ROOT / "scripts" / "frozen_vision"
for path in (str(EVAL_SCRIPTS), str(FVE_SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

detail_splits = importlib.import_module("prepare_detail_splits")
export_checkpoint = importlib.import_module("export_fve_checkpoint")
detail_runner = importlib.import_module("run_detail_benchmarks")
package_results = importlib.import_module("package_final_results")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_score(path: Path, fields: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(["split", *fields]) + "\n" + ",".join(["none", *map(str, fields.values())]) + "\n")


class DetailSplitTest(unittest.TestCase):
    def test_self_attested_one_row_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout_root = root / "heldout"
            heldout_root.mkdir()
            datasets = {}
            for logical, custom in detail_runner.DETAIL_DATASETS.items():
                source = root / f"source-{logical}.tsv"
                heldout = heldout_root / f"{custom}.tsv"
                source.write_text("index\tcategory\n1\tfake\n")
                heldout.write_text(source.read_text())
                datasets[logical] = {
                    "heldout_tsv": str(heldout.resolve()),
                    "heldout_tsv_bytes": heldout.stat().st_size,
                    "heldout_tsv_sha256": sha256(heldout),
                    "source": {
                        "bytes": source.stat().st_size,
                        "md5": "self-attested",
                        "path": str(source.resolve()),
                        "sha256": sha256(source),
                    },
                }
            manifest = root / "detail_split_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "datasets": datasets,
                        "generator_script_sha256": sha256(Path(detail_splits.__file__)),
                        "protocol": {"heldout_fraction": 0.4, "schema_version": 1, "seed": 42},
                        "split_id": "a" * 64,
                    }
                )
            )
            with self.assertRaises(SystemExit):
                detail_runner.split_fingerprint(manifest)

    def test_heldout_files_use_custom_dataset_names(self):
        self.assertEqual(
            set(detail_runner.DETAIL_CONFIG),
            {"MMBench_DEV_EN_V11_FVE_HELDOUT", "SEEDBench_IMG_FVE_HELDOUT"},
        )
        self.assertTrue(all(spec["class"] == "CustomMCQDataset" for spec in detail_runner.DETAIL_CONFIG.values()))

    def test_mmbench_circular_groups_never_cross_split(self):
        rows = []
        categories = sorted(detail_splits.MMBENCH_L2_CATEGORIES)
        for base in range(detail_splits.MMBENCH_ROWS // 4):
            category = categories[base % len(categories)]
            for offset in range(4):
                rows.append({"index": base + offset * 1_000_000, "l2-category": category})
        frame = pd.DataFrame(rows)
        heldout, info = detail_splits.split_mmbench(frame, seed=42)
        selected = set(heldout["index"].astype(int) % 1_000_000)
        self.assertEqual(len(heldout), len(selected) * 4)
        self.assertAlmostEqual(len(heldout) / len(frame), 0.4, places=2)
        for base, group in frame.assign(base=frame["index"] % 1_000_000).groupby("base"):
            self.assertIn(len(group[group["index"].isin(heldout["index"]) ]), (0, 4), base)
        self.assertIn("heldout_indices_sha256", info)

    def test_seed_split_is_deterministic_and_stratified(self):
        frame = pd.DataFrame(
            [
                {"index": index, "category": sorted(detail_splits.SEED_CATEGORIES)[index % 9]}
                for index in range(detail_splits.SEED_ROWS)
            ]
        )
        first, first_info = detail_splits.split_seed(frame, seed=42)
        second, second_info = detail_splits.split_seed(frame, seed=42)
        self.assertEqual(first["index"].tolist(), second["index"].tolist())
        self.assertEqual(first_info["heldout_indices_sha256"], second_info["heldout_indices_sha256"])
        self.assertAlmostEqual(len(first) / len(frame), 0.4, places=3)


class CommittedExportTest(unittest.TestCase):
    def test_evaluation_fingerprint_rehashes_exported_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_text("{}")
            shard = model / "model.safetensors"
            shard.write_bytes(b"first")
            (model / "fve_export_manifest.json").write_text(
                json.dumps(
                    {
                        "audit": {
                            "safetensor_sha256": {shard.name: sha256(shard)},
                            "support_file_sha256": {"config.json": sha256(model / "config.json")},
                        }
                    }
                )
            )
            run_six = importlib.import_module("run_six_benchmarks")
            self.assertEqual(run_six.model_fingerprint(model)["safetensor_sha256"][shard.name], sha256(shard))
            shard.write_bytes(b"other")
            with self.assertRaises(SystemExit):
                run_six.model_fingerprint(model)

    def test_tracker_wins_over_newer_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            committed = root / "global_step_1"
            actor = committed / "actor"
            (actor / "huggingface").mkdir(parents=True)
            (actor / "huggingface" / "config.json").write_text("{}")
            data = committed / "data.pt"
            data.write_bytes(b"data")
            (committed / "checkpoint_complete.json").write_text(
                json.dumps(
                    {
                        "actor_files": [],
                        "critic_files": [],
                        "global_step": 1,
                        "schema_version": 2,
                        "dataloader": {"bytes": 4, "path": "data.pt", "sha256": sha256(data)},
                    }
                )
            )
            (root / "global_step_2" / "actor").mkdir(parents=True)
            (root / "latest_checkpointed_iteration.txt").write_text("1")
            self.assertEqual(export_checkpoint.resolve_actor_dir(root), actor.resolve())


class PromptProtocolTest(unittest.TestCase):
    def test_mcq_prompt_patch_is_idempotent_and_pixel_budget_is_corrected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "image_mcq.py"
            target.write_text(
                "# Unrelated pinned MMMU-Pro branch:\n"
                "UNRELATED = 'The last line of your response should be of the following format:'\n"
                "            prompt += 'Please select the correct answer from the options above. \\n'\n"
            )
            command = [sys.executable, str(EVAL_SCRIPTS / "patch_vlmeval_mcq_prompt.py"), str(target)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = target.read_text()
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(target.read_text(), first)
            self.assertIn("<think>step-by-step reasoning</think> <answer>$LETTER</answer>", first)
        runner = (EVAL_SCRIPTS / "run_six_benchmarks.py").read_text()
        self.assertIn('"min_pixels": 1280 * 28 * 28', runner)
        self.assertNotIn('"min_pixels": 35840', runner)


class TableLineageTest(unittest.TestCase):
    def build_manifest(self, root: Path, role: str, detail: bool) -> Path:
        label = f"{role}-{'detail' if detail else 'full'}"
        score_dir = root / label
        mm_fields = {key: 0.5 for key in ("LR", "RR", "AR", "CP", "FP-C", "FP-S", "Overall")}
        seed_fields = {
            key: 0.6
            for key in (
                "Text Understanding",
                "Visual Reasoning",
                "Scene Understanding",
                "Instance Identity",
                "Instance Interaction",
                "Instance Attributes",
                "Instance Location",
                "Spatial Relation",
                "Instances Counting",
                "Overall",
            )
        }
        artifacts = {}
        write_score(score_dir / "mm.csv", mm_fields)
        write_score(score_dir / "seed.csv", seed_fields)
        artifacts.update({"MMBench_DEV_EN_V11": score_dir / "mm.csv", "SEEDBench_IMG": score_dir / "seed.csv"})
        if not detail:
            for name, value in (("VStarBench", 0.7), ("RealWorldQA", 0.8), ("BLINK", 0.9)):
                write_score(score_dir / f"{name}.csv", {"Overall": value})
                artifacts[name] = score_dir / f"{name}.csv"
            mme = score_dir / "mme.json"
            mme.write_text(json.dumps({"Overall": 0.4}))
            artifacts["MME-RealWorld-Lite"] = mme
        model_config = {
            "class": "Qwen2VLChat",
            "model_path": f"/{role}",
            "temperature": 0.01,
            "top_p": 0.001,
            "top_k": 1,
        }
        manifest = {
            "benchmarks": list(artifacts),
            "evaluation_role": "detail_40_percent_heldout" if detail else "full_six_benchmarks",
            "model_name": role.title(),
            "model_role": role,
            "package_freeze": ["fixture==1.0"],
            "prediction_artifacts": {},
            "provenance": {
                "config": {"data": {}, "model": {role: model_config}},
                "data": {"same": True},
                "data_manifest_sha256": "data-hash",
                "evaluator": {"same": True},
                "model": {"role": role},
                "package_freeze": ["fixture==1.0"],
                "split": {"split_id": "split"} if detail else None,
            },
            "run_id": "run-1",
            "schema_version": 1 if detail else 2,
            "score_artifact_sha256": {name: sha256(path) for name, path in artifacts.items()},
            "score_artifacts": {name: str(path) for name, path in artifacts.items()},
        }
        for name in artifacts:
            prediction = score_dir / f"prediction-{name}.xlsx"
            prediction.write_bytes(f"prediction:{role}:{name}".encode())
            manifest["prediction_artifacts"][name] = {
                "path": str(prediction),
                "rows": 1,
                "sha256": sha256(prediction),
                "sorted_index_sha256": "c" * 64,
            }
        if detail:
            manifest.update({"split_id": "a" * 64, "split_manifest_sha256": "b" * 64})
        manifest["score_input_prediction_sha256"] = {
            name: record["sha256"] for name, record in manifest["prediction_artifacts"].items()
        }
        manifest["attempt_fingerprint"] = hashlib.sha256(
            json.dumps(manifest["provenance"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path = score_dir / "manifest.json"
        (score_dir / "attempt_preflight.json").write_text(
            json.dumps(
                {"fingerprint": manifest["attempt_fingerprint"], "provenance": manifest["provenance"]}
            )
        )
        (score_dir / "vlmeval_config.json").write_text(json.dumps(manifest["provenance"]["config"]))
        path.write_text(json.dumps(manifest))
        return path

    def test_unpaired_detail_manifest_is_rejected(self):
        result = subprocess.run(
            [
                sys.executable,
                str(EVAL_SCRIPTS / "aggregate_paper_tables.py"),
                "--base-manifest",
                "missing",
                "--rotation-manifest",
                "missing",
                "--base-detail-manifest",
                "missing",
                "--output-dir",
                "missing",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("both detail manifests or neither", result.stderr)

    def test_compatible_pair_produces_tables_and_explicit_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.build_manifest(root, "base", detail=False)
            rotation = self.build_manifest(root, "rotation", detail=False)
            base_detail = self.build_manifest(root, "base", detail=True)
            rotation_detail = self.build_manifest(root, "rotation", detail=True)
            output = root / "tables"
            result = subprocess.run(
                [
                    sys.executable,
                    str(EVAL_SCRIPTS / "aggregate_paper_tables.py"),
                    "--base-manifest",
                    str(base),
                    "--rotation-manifest",
                    str(rotation),
                    "--base-detail-manifest",
                    str(base_detail),
                    "--rotation-detail-manifest",
                    str(rotation_detail),
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            tables = json.loads((output / "paper_tables.json").read_text())
            self.assertEqual(tables["tables"]["table_i"]["rows"][0]["MME"], "40.00")
            self.assertEqual(tables["detail_protocol"], "deterministic 40% held-out split")


class PackageLineageTest(unittest.TestCase):
    @staticmethod
    def record(path: Path) -> dict:
        return {"bytes": path.stat().st_size, "path": str(path.resolve()), "sha256": sha256(path)}

    def test_complete_synthetic_lineage_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = TableLineageTest()
            base_manifest = helper.build_manifest(root, "base", detail=False)
            rotation_manifest = helper.build_manifest(root, "rotation", detail=False)
            split_manifest = root / "detail_split_manifest.json"
            split_manifest.write_text(json.dumps({"split_id": "a" * 64}))
            base_detail = helper.build_manifest(root, "base", detail=True)
            rotation_detail = helper.build_manifest(root, "rotation", detail=True)
            for detail_path in (base_detail, rotation_detail):
                detail = json.loads(detail_path.read_text())
                detail["split_manifest"] = str(split_manifest.resolve())
                detail["split_manifest_sha256"] = sha256(split_manifest)
                detail_path.write_text(json.dumps(detail))
            tables = root / "tables"
            aggregate = subprocess.run(
                [
                    sys.executable,
                    str(EVAL_SCRIPTS / "aggregate_paper_tables.py"),
                    "--base-manifest",
                    str(base_manifest),
                    "--rotation-manifest",
                    str(rotation_manifest),
                    "--base-detail-manifest",
                    str(base_detail),
                    "--rotation-detail-manifest",
                    str(rotation_detail),
                    "--output-dir",
                    str(tables),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(aggregate.returncode, 0, aggregate.stderr)

            training_dir = root / "runs" / "run-1"
            train_run = training_dir / "experiment"
            checkpoint = train_run / "checkpoints" / "global_step_20" / "checkpoint_complete.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({"global_step": 20}))
            summary = train_run / "reward_traces" / "rotation" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"trace_validated": True}))
            stage_tracker = train_run / "checkpoints" / "checkpoint_stage.train.json"
            stage_tracker.write_text(json.dumps({"global_step": 20, "mode": "train"}))
            difficulty = train_run / "reward_analysis" / "rotation.train" / "difficulty_analysis_manifest.json"
            difficulty.parent.mkdir(parents=True)
            difficulty_outputs = {}
            for name in (
                "reward_difficulty_by_stage.csv",
                "reward_difficulty_by_step.csv",
                "reward_difficulty_groups.parquet",
                "reward_difficulty_summary.json",
            ):
                output = difficulty.parent / name
                output.write_text(f"synthetic {name}\n")
                difficulty_outputs[name] = sha256(output)
            difficulty.write_text(
                json.dumps(
                    {
                        "analysis_fingerprint": "d" * 64,
                        "output_sha256": difficulty_outputs,
                        "run_id": "run-1",
                        "schema_version": 1,
                        "task": "rotation",
                    }
                )
            )

            data_manifest = root / "rotation-data.json"
            data_manifest.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "files": {"train.parquet": {}, "test.parquet": {}},
                        "split_mapping": {"test": "train.parquet", "validation": "test.parquet"},
                    }
                )
            )
            queue_manifest = root / "queue.complete"
            queue_manifest.write_text(
                json.dumps(
                    {
                        "invocation_provenance": {
                            "data_manifest": self.record(data_manifest),
                            "environment": {"validated": True},
                        },
                        "run_id": "run-1",
                        "schema_version": 1,
                        "stages": [
                            {
                                "checkpoint": {
                                    "completion_manifest": self.record(checkpoint),
                                    "tracker": self.record(stage_tracker),
                                },
                                "difficulty_analysis": self.record(difficulty),
                                "mode": mode,
                                "reward_summary": self.record(summary),
                                "run_dir": str(train_run.resolve()),
                                "task": "rotation",
                            }
                            for mode in ("quick", "quick-resume", "production-smoke")
                        ]
                        + [
                            {
                                "checkpoint": {
                                    "completion_manifest": self.record(checkpoint),
                                    "global_step": 20,
                                    "tracker": self.record(stage_tracker),
                                },
                                "difficulty_analysis": self.record(difficulty),
                                "mode": "train",
                                "reward_summary": self.record(summary),
                                "run_dir": str(train_run.resolve()),
                                "task": "rotation",
                            },
                        ],
                        "status": "complete",
                    }
                )
            )

            exported = root / "exported"
            exported.mkdir()
            export_manifest = exported / "fve_export_manifest.json"
            exported_shard = exported / "model.safetensors"
            exported_shard.write_bytes(b"synthetic weights")
            export_manifest.write_text(
                json.dumps(
                    {
                        "audit": {"safetensor_sha256": {exported_shard.name: sha256(exported_shard)}},
                        "global_step": 20,
                    }
                )
            )
            export_hash = sha256(export_manifest)
            export_weights = {exported_shard.name: sha256(exported_shard)}
            post_dir = root / "post"
            post_dir.mkdir()
            smoke_dir = post_dir / "mmbench_smoke"
            smoke_dir.mkdir()
            smoke_payload = {
                "model_path": str(exported.resolve()),
                "sample_count": 24,
                "table2_smoke_percent": {
                    name: 50.0
                    for name in (
                        "Logical",
                        "Relation",
                        "Attribute",
                        "Coarse",
                        "Cross-Inst.",
                        "Single-Inst.",
                        "Average",
                    )
                },
            }
            smoke_manifest = smoke_dir / "smoke_manifest.json"
            smoke_manifest.write_text(json.dumps(smoke_payload))
            smoke_table = smoke_dir / "mmbench_v11_smoke_table2.json"
            smoke_table.write_text(json.dumps(smoke_payload["table2_smoke_percent"]))
            smoke_scores = smoke_dir / "mmbench_v11_smoke_official_scores.csv"
            smoke_scores.write_text("Overall\n1.0\n")
            smoke_predictions = smoke_dir / "mmbench_v11_smoke_predictions.xlsx"
            smoke_predictions.write_bytes(b"synthetic smoke predictions")
            smoke_subset = smoke_dir / "mmbench_v11_smoke_subset.tsv"
            smoke_subset.write_text("index\n1\n")
            smoke_log = post_dir / "mmbench_smoke.log"
            smoke_log.write_text("synthetic smoke\n")
            post_manifest = post_dir / "post_training_manifest.json"
            post_manifest.write_text(
                json.dumps(
                    {
                        "export": {"exported_model": str(exported)},
                        "export_manifest_sha256": export_hash,
                        "mmbench_smoke": smoke_payload,
                        "mmbench_smoke_manifest_sha256": sha256(smoke_manifest),
                        "model_role": "rotation",
                        "provenance": {"checkpoint_manifest_sha256": sha256(checkpoint)},
                        "run_id": "run-1",
                        "schema_version": 2,
                        "smoke_artifacts": {
                            path.name: self.record(path)
                            for path in (
                                smoke_manifest,
                                smoke_table,
                                smoke_scores,
                                smoke_predictions,
                                smoke_subset,
                            )
                        },
                        "smoke_log": self.record(smoke_log),
                    }
                )
            )
            rotation = json.loads(rotation_manifest.read_text())
            rotation["provenance"]["model"]["export_manifest_sha256"] = export_hash
            rotation["provenance"]["model"]["safetensor_sha256"] = export_weights
            rotation["provenance"]["model"]["support_file_sha256"] = {}
            rotation["model_path"] = str(exported.resolve())
            rotation["attempt_fingerprint"] = hashlib.sha256(
                json.dumps(rotation["provenance"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            rotation_manifest.write_text(json.dumps(rotation))
            (rotation_manifest.parent / "attempt_preflight.json").write_text(
                json.dumps(
                    {"fingerprint": rotation["attempt_fingerprint"], "provenance": rotation["provenance"]}
                )
            )
            rotation_detail_payload = json.loads(rotation_detail.read_text())
            rotation_detail_payload["provenance"]["model"] = dict(rotation["provenance"]["model"])
            rotation_detail_payload["attempt_fingerprint"] = hashlib.sha256(
                json.dumps(rotation_detail_payload["provenance"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            rotation_detail.write_text(json.dumps(rotation_detail_payload))
            (rotation_detail.parent / "attempt_preflight.json").write_text(
                json.dumps(
                    {
                        "fingerprint": rotation_detail_payload["attempt_fingerprint"],
                        "provenance": rotation_detail_payload["provenance"],
                    }
                )
            )
            # Regenerate tables so their input hash binds the updated Rotation manifest.
            aggregate = subprocess.run(
                [
                    sys.executable,
                    str(EVAL_SCRIPTS / "aggregate_paper_tables.py"),
                    "--base-manifest",
                    str(base_manifest),
                    "--rotation-manifest",
                    str(rotation_manifest),
                    "--base-detail-manifest",
                    str(base_detail),
                    "--rotation-detail-manifest",
                    str(rotation_detail),
                    "--output-dir",
                    str(tables),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(aggregate.returncode, 0, aggregate.stderr)

            final = root / "final"
            package_argv = [
                    str(EVAL_SCRIPTS / "package_final_results.py"),
                    "--run-id",
                    "run-1",
                    "--training-run-dir",
                    str(training_dir),
                    "--queue-manifest",
                    str(queue_manifest),
                    "--post-training-manifest",
                    str(post_manifest),
                    "--rotation-data-manifest",
                    str(data_manifest),
                    "--base-eval-manifest",
                    str(base_manifest),
                    "--rotation-eval-manifest",
                    str(rotation_manifest),
                    "--tables-dir",
                    str(tables),
                    "--output-dir",
                    str(final),
                ]
            with (
                mock.patch.object(sys, "argv", package_argv),
                mock.patch.object(
                    package_results,
                    "split_fingerprint",
                    return_value=({"split_id": "a" * 64}, {"verified": True}),
                ),
                redirect_stdout(io.StringIO()),
            ):
                package_results.main()
            archive = final / "fve_run-1_results.tar.gz"
            self.assertTrue(archive.is_file())
            self.assertIn(sha256(archive), (archive.with_suffix(archive.suffix + ".sha256")).read_text())
            with tarfile.open(archive) as handle:
                self.assertIn("fve_run-1_results/workflow/queue.complete.json", handle.getnames())


if __name__ == "__main__":
    unittest.main()
