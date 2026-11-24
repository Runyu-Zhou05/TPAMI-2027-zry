#!/usr/bin/env python3
"""Launch one normal (unfrozen) Table-XI harder-variant training run."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv"
)


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    limit_images: int
    experiment: str


TASKS = {
    "position3x3": TaskSpec(
        "MMBench-PositionQA_3x3", 2, "ssl4rl_qwen_3b_mmbench_position3x3"
    ),
    "rotation45": TaskSpec(
        "MMBench_RotationQA_45degree", 2, "ssl4rl_qwen_3b_mmbench_rotation45"
    ),
    "jigsaw3x3": TaskSpec(
        "MMBench_JigsawQA_3x3", 9, "ssl4rl_qwen_3b_mmbench_jigsaw3x3"
    ),
    "contrastive_strong": TaskSpec(
        "MMBench-ContrastiveQA_strong", 2, "ssl4rl_qwen_3b_mmbench_contrastive_strong"
    ),
}


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise SystemExit(
            "--run-id must contain only letters, digits, '.', '_' or '-' and be at most 80 characters"
        )
    return value


def parse_gpus(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if (
        len(gpu_ids) != 8
        or len(set(gpu_ids)) != 8
        or any(not item.isdigit() for item in gpu_ids)
    ):
        raise SystemExit("Table-XI training requires exactly eight unique numeric GPU indices")
    return gpu_ids


def require_input(path: Path, description: str, *, directory: bool = False) -> None:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise SystemExit(f"Missing {description} {kind}: {path}")


def hydra_files(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{path}'" for path in paths) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Exactly eight comma-separated physical GPU indices",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("HARD_MODEL_PATH", ROOT / "models/Qwen2.5-VL-3B-Instruct")),
        help="Base Qwen2.5-VL-3B-Instruct checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("HARD_DATA_ROOT", ROOT / "our_datasets")),
        help="Parent of the four preprocessed hard-variant dataset directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("HARD_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
    )
    parser.add_argument(
        "--python", dest="python_executable", default=os.environ.get("HARD_PYTHON", sys.executable)
    )
    parser.add_argument(
        "--reward-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record per-rollout reward traces (enabled by default)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = validate_run_id(args.run_id)
    gpu_ids = parse_gpus(args.gpus)
    spec = TASKS[args.task]
    model_path = args.model_path.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_dir = data_root / spec.dataset
    train_file = dataset_dir / "train.parquet"
    test_file = dataset_dir / "test.parquet"

    # Keep smoke artifacts isolated from the production directory even when a
    # caller deliberately reuses the same run ID for both gates.
    experiment = spec.experiment if args.mode == "train" else f"{spec.experiment}_smoke"
    run_dir = output_root / "runs" / run_id / experiment
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    trace_root = run_dir / "reward_traces"
    ray_tmp = Path("/tmp/hard-mmbench-ray") / f"{run_id}-{args.task}-{args.mode}"
    hf_home = output_root / "cache" / "huggingface"

    if not args.dry_run:
        require_input(model_path, "base model", directory=True)
        require_input(train_file, "training Parquet")
        require_input(test_file, "test Parquet")
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"Refusing to reuse nonempty hard-variant run: {run_dir}")
        for directory in (checkpoint_dir, tensorboard_dir, ray_tmp, hf_home):
            directory.mkdir(parents=True, exist_ok=True)
        if args.reward_trace:
            trace_root.mkdir(parents=True, exist_ok=True)

    train_batch_size = 512
    ppo_mini_batch_size = 128
    ppo_micro_batch_size = 10
    max_response_length = 2048
    total_epochs = 20
    total_training_steps = "null"
    save_freq = 50
    test_freq = 200
    val_before_train = "True"
    if args.mode == "smoke":
        # Eight prompts (one per GPU) and n=5 exercise the real multi-rollout
        # and optimizer path while keeping the gate to exactly one update.
        train_batch_size = 8
        ppo_mini_batch_size = 8
        ppo_micro_batch_size = 1
        max_response_length = 128
        total_epochs = 1
        total_training_steps = "1"
        save_freq = 1
        test_freq = -1
        val_before_train = "False"

    command = [
        args.python_executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        # This intentionally reproduces the paper launcher: both generated
        # splits optimize the policy, and test.parquet is also the val split.
        f"data.train_files={hydra_files([train_file, test_file])}",
        f"data.val_files={hydra_files([test_file])}",
        f"data.train_batch_size={train_batch_size}",
        "+data.seed=42",
        "data.max_prompt_length=1024",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.image_key=images",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.freeze_vision_encoder=False",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size}",
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
        "+actor_rollout_ref.rollout.limit_images=" + str(spec.limit_images),
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console','tensorboard']",
        "trainer.project_name=ssl4rl_mmbench_hard",
        f"trainer.experiment_name={experiment}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.val_before_train={val_before_train}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.total_training_steps={total_training_steps}",
        "trainer.resume_mode=disable",
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    if args.reward_trace:
        command.extend(
            [
                f"trainer.reward_trace_dir={trace_root}",
                f"trainer.reward_trace_task={args.task}",
                f"trainer.reward_trace_run_id={run_id}",
            ]
        )

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
    print(
        f"Table-XI task={args.task} mode={args.mode} run_id={run_id} "
        f"physical_gpus={gpu_ids}"
    )
    print(f"Output directory: {run_dir}")
    print(
        f"Global rollout batch={train_batch_size * 5}; "
        f"global PPO mini-batch={ppo_mini_batch_size * 5}"
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)


if __name__ == "__main__":
    main()
