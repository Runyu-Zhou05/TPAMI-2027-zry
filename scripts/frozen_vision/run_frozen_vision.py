#!/usr/bin/env python3
"""Launch one Frozen Vision Encoder (FVE) SSL4RL task portably."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from fve_runtime import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAY_TMP_ROOT,
    ROOT,
    TASKS,
    default_run_id,
    resolve_path,
    short_ray_tmpdir,
    stage_layout,
    validate_run_id,
)


# These values preserve the original global batch: 512 prompts x 5 rollouts =
# 2560 responses, with a global PPO mini-batch of 128 x 5 = 640 responses.
MICRO_BATCH = {1: 10, 2: 10, 4: 10, 5: 8, 8: 10, 10: 8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU indices")
    parser.add_argument("--mode", choices=["quick", "quick-resume", "production-smoke", "train"], required=True)
    parser.add_argument("--output-root", default=os.environ.get("FVE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--run-id", default=os.environ.get("FVE_RUN_ID"))
    parser.add_argument("--model-path", default=os.environ.get("FVE_MODEL_PATH"))
    parser.add_argument("--data-root", default=os.environ.get("FVE_DATA_ROOT"))
    parser.add_argument("--smoke-data-root", default=os.environ.get("FVE_SMOKE_DATA_ROOT"))
    parser.add_argument("--python", dest="python_executable", default=os.environ.get("FVE_PYTHON", sys.executable))
    parser.add_argument("--ray-tmp-root", default=os.environ.get("FVE_RAY_TMP_ROOT", str(DEFAULT_RAY_TMP_ROOT)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    run_id = validate_run_id(args.run_id or default_run_id("manual-fve"))
    data_root = resolve_path(args.data_root or ROOT / "our_datasets")
    smoke_data_root = resolve_path(args.smoke_data_root or output_root / "smoke_data")
    ray_tmp_root = resolve_path(args.ray_tmp_root)
    if args.model_path is None:
        if not args.dry_run:
            raise SystemExit("--model-path (or FVE_MODEL_PATH) is required for a real launch")
        model_path = "<set --model-path or FVE_MODEL_PATH>"
    else:
        model_path = str(resolve_path(args.model_path))

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    gpu_count = len(gpu_ids)
    if gpu_count not in MICRO_BATCH:
        raise SystemExit(f"Exact effective batch supports GPU counts {sorted(MICRO_BATCH)}, got {gpu_count}")
    if len(set(gpu_ids)) != gpu_count:
        raise SystemExit("GPU indices must be unique")

    dataset_dir, _ = TASKS[args.task]
    micro_batch = MICRO_BATCH[gpu_count]
    if 640 % gpu_count or (640 // gpu_count) % micro_batch:
        raise SystemExit("Internal batch-size invariant failed")

    train_file = data_root / dataset_dir / "train.parquet"
    test_file = data_root / dataset_dir / "test.parquet"
    train_files = f"['{train_file}','{test_file}']"
    rollout_n = 5
    train_batch_size = 512
    ppo_mini_batch_size = 128
    max_response_length = 2048
    total_training_steps = "null"
    total_epochs = 20
    save_freq = 200
    test_freq = 200
    val_before_train = "True"
    resume_mode = "auto"

    if args.mode in {"quick", "quick-resume"}:
        smoke = smoke_data_root / "quick.parquet"
        train_files = f"['{smoke}']"
        test_file = smoke
        rollout_n = 2
        # Keep the smoke gate tiny but make both the rollout and PPO mini-batch
        # divisible by every supported GPU count, including eight H20s.
        train_batch_size = gpu_count
        ppo_mini_batch_size = gpu_count
        micro_batch = 1
        max_response_length = 128
        total_training_steps = "1" if args.mode == "quick" else "2"
        total_epochs = 2
        save_freq = 1
        test_freq = -1
        val_before_train = "False"
        resume_mode = "disable" if args.mode == "quick" else "auto"
    elif args.mode == "production-smoke":
        smoke = smoke_data_root / "production_shape.parquet"
        train_files = f"['{smoke}']"
        test_file = smoke_data_root / "quick.parquet"
        total_training_steps = "1"
        total_epochs = 1
        save_freq = 1
        test_freq = -1
        val_before_train = "False"
        resume_mode = "disable"

    experiment, run_dir, trace_task = stage_layout(output_root, run_id, args.task, args.mode)
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    trace_root = run_dir / "reward_traces"
    ray_tmp = short_ray_tmpdir(ray_tmp_root, run_id, experiment)
    hf_home = output_root / "cache" / "huggingface"
    if not args.dry_run:
        for directory in (checkpoint_dir, tensorboard_dir, trace_root, ray_tmp, hf_home):
            directory.mkdir(parents=True, exist_ok=True)

    command = [
        args.python_executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_files}",
        f"data.val_files=['{test_file}']",
        f"data.train_batch_size={train_batch_size}",
        "+data.seed=42",
        "data.max_prompt_length=1024",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.image_key=images",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.freeze_vision_encoder=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console','tensorboard']",
        "trainer.project_name=ssl4rl_frozen_vision",
        f"trainer.experiment_name={experiment}",
        f"trainer.n_gpus_per_node={gpu_count}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.val_before_train={val_before_train}",
        f"trainer.default_local_dir={checkpoint_dir}",
        "trainer.max_actor_ckpt_to_keep=2",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.total_training_steps={total_training_steps}",
        f"trainer.resume_mode={resume_mode}",
        f"trainer.reward_trace_dir={trace_root}",
        f"trainer.reward_trace_task={trace_task}",
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
    print(f"Launching FVE {args.task} mode={args.mode} run_id={run_id} physical_gpus={gpu_ids}")
    print(f"Output directory: {run_dir}")
    print(f"Global rollout batch={train_batch_size * rollout_n}; global PPO mini-batch={ppo_mini_batch_size * rollout_n}")
    print(f"Per-GPU PPO mini-batch={(ppo_mini_batch_size * rollout_n) // gpu_count}; micro={micro_batch}")
    sys.stdout.flush()
    if args.dry_run:
        print(" ".join(command))
        return
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)


if __name__ == "__main__":
    main()
