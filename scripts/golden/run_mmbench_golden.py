#!/usr/bin/env python3
"""Launch the Position-initialized MMBench Golden-RLVR stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "scripts" / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
from run_six_benchmarks import model_fingerprint  # noqa: E402
from prepare_mmbench_golden_data import digest_file, validate_parquet  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv")
DATASET_NAME = "MMBench_GoldenQA"
EXPECTED_ROWS = {"train.parquet": 2_607, "test.parquet": 1_722}
EXPECTED_EPOCHS = 20
EXPECTED_STEPS_PER_EPOCH = 5
EXPECTED_TOTAL_STEPS = 100
SAVE_TEST_FREQUENCY = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise SystemExit("--run-id must contain only letters, digits, '.', '_' or '-' and be at most 80 characters")
    return value


def validate_position_export(model_path: Path) -> dict:
    model_path = model_path.expanduser().resolve()
    record = model_fingerprint(model_path)
    if record.get("artifact_type") != "ssl4rl_export" or record.get("model_role") != "position":
        raise SystemExit("Golden stage requires a validated normal Position ssl4rl_export")
    export_path = Path(record["export_manifest"])
    export = json.loads(export_path.read_text())
    actor_checkpoint = str(export.get("actor_checkpoint", ""))
    if "position" not in actor_checkpoint.lower() or "frozen_vision" in actor_checkpoint.lower():
        raise SystemExit(f"Golden stage export is not bound to a normal Position checkpoint: {actor_checkpoint}")
    return record


def validate_golden_data(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Golden data manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_protocol = {
        "heldout_fraction": 0.4,
        "schema_version": 1,
        "seed": 42,
        "source_md5": "b6caf1133a01c6bb705cf753bb527ed8",
    }
    split = manifest.get("split", {})
    train_indices = split.get("train_indices", [])
    test_indices = split.get("test_indices", [])
    generator = Path(__file__).with_name("prepare_mmbench_golden_data.py")
    source_path = Path(manifest.get("source", {}).get("path", "")).expanduser().resolve()
    if (
        manifest.get("dataset_name") != DATASET_NAME
        or manifest.get("protocol") != expected_protocol
        or manifest.get("source", {}).get("md5") != expected_protocol["source_md5"]
        or len(train_indices) != EXPECTED_ROWS["train.parquet"]
        or len(test_indices) != EXPECTED_ROWS["test.parquet"]
        or len(set(train_indices) | set(test_indices)) != sum(EXPECTED_ROWS.values())
        or set(train_indices) & set(test_indices)
        or split.get("train_indices_sha256")
        != hashlib.sha256(("\n".join(train_indices) + "\n").encode()).hexdigest()
        or split.get("test_indices_sha256")
        != hashlib.sha256(("\n".join(test_indices) + "\n").encode()).hexdigest()
        or manifest.get("generator_script_sha256") != sha256_file(generator)
        or not source_path.is_file()
        or digest_file(source_path, "md5") != expected_protocol["source_md5"]
    ):
        raise SystemExit(f"Golden data protocol mismatch: {manifest_path}")
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = dataset_dir / filename
        record = manifest.get("outputs", {}).get(filename, {})
        actual = validate_parquet(path, expected_rows, "train" if filename.startswith("train") else "test")
        if (
            record.get("rows") != expected_rows
            or actual != record
        ):
            raise SystemExit(f"Golden data artifact is missing or changed: {path}")
    return {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}


def validate_completed_run(checkpoint_dir: Path, validation_dir: Path, run_id: str) -> dict:
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text().strip() != str(EXPECTED_TOTAL_STEPS):
        raise SystemExit(f"Golden run did not commit terminal step {EXPECTED_TOTAL_STEPS}: {tracker}")
    completion = checkpoint_dir / f"global_step_{EXPECTED_TOTAL_STEPS}" / "checkpoint_complete.json"
    if not completion.is_file():
        raise SystemExit(f"Golden terminal checkpoint manifest is missing: {completion}")
    payload = json.loads(completion.read_text())
    if (
        payload.get("schema_version") != 2
        or int(payload.get("global_step", -1)) != EXPECTED_TOTAL_STEPS
        or payload.get("run_id") != run_id
    ):
        raise SystemExit(f"Golden terminal checkpoint identity mismatch: {completion}")
    expected_validation = [0, 20, 40, 60, 80, 100]
    missing = [step for step in expected_validation if not (validation_dir / f"{step}.jsonl").is_file()]
    if missing:
        raise SystemExit(f"Golden validation dumps are incomplete; missing steps={missing}")
    return {
        "checkpoint_manifest": str(completion),
        "checkpoint_manifest_sha256": sha256_file(completion),
        "global_step": EXPECTED_TOTAL_STEPS,
        "validation_steps": expected_validation,
    }


def short_ray_tmpdir(run_id: str) -> Path:
    token = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return Path("/tmp/golden-ray") / token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="Exported final Position HF model")
    parser.add_argument("--data-root", type=Path, required=True, help="Parent containing MMBench_GoldenQA")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("FVE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", required=True, help="Exactly eight comma-separated physical GPU indices")
    parser.add_argument("--python", dest="python_executable", default=os.environ.get("FVE_PYTHON", sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = validate_run_id(args.run_id)
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or any(not value.isdigit() for value in gpu_ids):
        raise SystemExit("Golden production training requires exactly eight unique GPU indices")
    model_path = args.model_path.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_dir = data_root / DATASET_NAME
    if not args.dry_run:
        model_identity = validate_position_export(model_path)
        data_identity = validate_golden_data(dataset_dir)
    else:
        model_identity = {"artifact_type": "ssl4rl_export", "model_role": "position"}
        data_identity = {"manifest": str(dataset_dir / "manifest.json")}

    experiment = "ssl4rl_qwen_3b_mmbench_position_to_golden"
    run_dir = output_root / "runs" / run_id / experiment
    checkpoint_dir = run_dir / "checkpoints"
    trace_root = run_dir / "reward_traces"
    tensorboard_dir = run_dir / "tensorboard"
    validation_dir = run_dir / "validation_generations"
    ray_tmp = short_ray_tmpdir(run_id)
    hf_home = output_root / "cache" / "huggingface"
    if not args.dry_run:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"Refusing to reuse nonempty Golden run: {run_dir}")
        for directory in (checkpoint_dir, trace_root, tensorboard_dir, ray_tmp, hf_home):
            directory.mkdir(parents=True, exist_ok=True)
        provenance = {
            "data": data_identity,
            "expected_checkpoints": [20, 40, 60, 80, 100],
            "model": model_identity,
            "protocol": {
                "epochs": EXPECTED_EPOCHS,
                "filter_overlong_prompts": False,
                "max_prompt_length": 1024,
                "save_freq": SAVE_TEST_FREQUENCY,
                "seed": 42,
                "test_freq": SAVE_TEST_FREQUENCY,
                "total_training_steps": EXPECTED_TOTAL_STEPS,
                "truncation": "error",
                "val_before_train": True,
            },
            "run_id": run_id,
        }
        (run_dir / "golden_preflight.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    train_file = dataset_dir / "train.parquet"
    test_file = dataset_dir / "test.parquet"
    command = [
        args.python_executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files=['{train_file}']",
        f"data.val_files=['{test_file}']",
        "data.train_batch_size=512",
        "+data.seed=42",
        "data.max_prompt_length=1024",
        "data.max_response_length=2048",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "data.image_key=images",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.freeze_vision_encoder=False",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=128",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console','tensorboard']",
        "trainer.project_name=ssl4rl_position_to_golden",
        f"trainer.experiment_name={experiment}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        f"trainer.save_freq={SAVE_TEST_FREQUENCY}",
        f"trainer.test_freq={SAVE_TEST_FREQUENCY}",
        "trainer.val_before_train=True",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.validation_data_dir={validation_dir}",
        "trainer.max_actor_ckpt_to_keep=null",
        f"trainer.total_epochs={EXPECTED_EPOCHS}",
        f"trainer.total_training_steps={EXPECTED_TOTAL_STEPS}",
        "trainer.resume_mode=disable",
        f"trainer.reward_trace_dir={trace_root}",
        "trainer.reward_trace_task=mmbench_golden",
        f"trainer.reward_trace_run_id={run_id}",
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "TENSORBOARD_DIR": str(tensorboard_dir),
            "RAY_TMPDIR": str(ray_tmp),
            "PYTHONHASHSEED": "42",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HOME": str(hf_home),
            "HF_HUB_DISABLE_XET": environment.get("HF_HUB_DISABLE_XET", "1"),
        }
    )
    print(f"Position -> Golden run_id={run_id} GPUs={gpu_ids}")
    print(f"Expected budget: {EXPECTED_EPOCHS} epochs x {EXPECTED_STEPS_PER_EPOCH} steps = {EXPECTED_TOTAL_STEPS}")
    print("Expected validation/checkpoints: step 0 validation; steps 20,40,60,80,100 validation+checkpoint")
    print(f"Output directory: {run_dir}")
    if args.dry_run:
        print(" ".join(command))
        return
    completed = subprocess.run(command, cwd=ROOT, env=environment)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    result = validate_completed_run(checkpoint_dir, validation_dir, run_id)
    (run_dir / "golden_complete.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
