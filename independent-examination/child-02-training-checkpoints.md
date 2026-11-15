# Independent examination: Rotation training, reward traces, resume, and checkpoints

Scope was static/offline inspection plus CPU-only configuration and resolver fixtures. No SSH, network, GPU, inference, dataset download, or training smoke run was used. The production code changes documented by the lead were not made by this child audit.

## Confirmed issues

### 1. High: traces written after the committed checkpoint make auto-resume fail

- **Evidence:** production checkpoints are every 200 steps (`scripts/frozen_vision/run_frozen_vision.py:74-77,167`); a step trace is published before actor update and checkpointing (`verl/trainer/ppo/ray_trainer.py:1303-1317,1354-1361,1395-1410`); an existing trace is a hard error (`verl/utils/reward_trace.py:84-89`).
- **Impact:** after rollback to checkpoint K, the rerun of K+1 collides with the trace left by the interrupted attempt. Before step 200, virtually any post-trace interruption makes `resume_mode=auto` restart at step 1 and fail on `step_00000001.parquet`.
- **Reproduction/static test:** tracker step 200 plus an existing `step_00000201.parquet` necessarily resumes at 201 and reaches the recorder's `FileExistsError` before the update is committed.
- **Recommended fix:** namespace traces by attempt or atomically reconcile/quarantine traces above the committed checkpoint on resume; retain overwrite protection at or below it and add an interruption/resume test.

### 2. High: resuming a terminal checkpoint performs one extra optimizer step

- **Evidence:** checkpoint load restores `global_steps` (`verl/trainer/ppo/ray_trainer.py:997-1052,1111-1114`), after which the trainer increments unconditionally (`:1126-1135`). A restored step equal to the total therefore enters the loop as total+1, updates the actor, saves, and only then returns (`:1181,1354-1361,1395-1445`).
- **Impact:** recovery after the final checkpoint changes the supposedly final model and exceeds the configured optimization budget.
- **Reproduction/static test:** with total 420 and tracker 420, the control flow executes and saves step 421.
- **Recommended fix:** after load, return success at equality and reject a checkpoint beyond the configured total unless explicit extension is requested.

### 3. High/protocol decision: the logical validation split is also optimized on

- **Evidence:** production `data.train_files` contains both `train.parquet` and `test.parquet`, while `data.val_files` uses `test.parquet` (`scripts/frozen_vision/run_frozen_vision.py:67-69,124-126`). The preparation code maps raw validation to logical test (`scripts/frozen_vision/prepare_rotation_data.py:329-342`). The original `run_mmbench_rotation.sh:4-13` has the same behavior.
- **Impact:** step validation is not held out. It may also overlap the later MMBench DEV evaluation. Because the original script does this too, the lead treats it as an unresolved transductive-protocol decision rather than silently changing it.
- **Reproduction/static test:** the 8-GPU dry run prints the test path in both train and validation lists.
- **Recommended fix:** use only logical train for optimization, or explicitly document and fingerprint the transductive protocol, rename the split, and never describe its validation metric as held out.

### 4. Medium-high: export can select an uncommitted partial checkpoint

- **Evidence:** `scripts/evaluation/export_fve_checkpoint.py:29-42` ignores `latest_checkpointed_iteration.txt` and chooses the numerically greatest `global_step_*` with an actor directory and HF config. Direct HF mode is selected when any safetensors file exists (`:132-143`).
- **Impact:** a terminated higher-numbered save masks the last committed checkpoint; an existing stale merged target can also be reused without source binding.
- **Reproduction/static test:** a fixture with committed step 200 and partial step 201 caused `resolve_actor_dir()` to choose step 201.
- **Recommended fix:** resolve via an atomically published committed-checkpoint manifest/tracker and validate the complete shard/index set before selection; bind any cached merged export to source hashes.

### 5. Medium: the checkpoint tracker is directly truncated and written

- **Evidence:** despite the “atomic usage” comment, `verl/trainer/ppo/ray_trainer.py:990-995` uses `open(..., "w")`. The reader directly parses the bytes as an integer without corruption recovery (`verl/utils/checkpoint/checkpoint_manager.py:167-196`).
- **Impact:** interruption can leave an empty/malformed tracker and block resume even with a complete checkpoint directory.
- **Reproduction/static test:** a zero-length tracker raises `ValueError` at `int(f.read().decode())`.
- **Recommended fix:** write, flush/fsync, and `os.replace` a sibling temporary file after checkpoint completeness validation; allow fallback only to directories carrying complete manifests.

### 6. Medium: required frozen-model counts are printed rather than enforced

- **Evidence:** the worker freezes `visual`, calculates visual/trainable counts, and only prints them (`verl/workers/fsdp_workers.py:330-345`). Exact total and visual counts are not checked until export (`scripts/evaluation/export_fve_checkpoint.py:86-89`).
- **Impact:** an incorrect Qwen2.5-VL model can consume an entire training run before rejection.
- **Recommended fix:** assert expected architecture and 668,684,288 visual / 3,085,938,688 trainable / 3,754,622,976 total parameters before FSDP and optimizer creation.

### 7. Medium: reward summaries do not prove trace completeness

- **Evidence:** `scripts/frozen_vision/summarize_rewards.py:17-38` reads only embedded step and reward columns and reports aggregates. It does not validate schema, filename-to-step agreement, contiguity, run/task identity, expected rows per step, or agreement with the final checkpoint.
- **Impact:** missing, mixed-run, or incomplete traces can still produce `summary.json` and allow queue completion.
- **Recommended fix:** validate the full schema, one run/task, contiguous steps, stage-specific response counts, filename equality, and the committed final step.

## Checked areas without a confirmed issue

- The 8-GPU production dry run preserves prompt batch 512, five rollouts, rollout batch 2,560, PPO mini-batch 640, per-GPU mini-batch 80, and micro-batch 10.
- `freeze_vision_encoder=True`, `use_orig_params=True`, and a trainable-only optimizer are connected.
- The first optimizer step audits no visual gradients, at least one language gradient, and an unchanged visual checksum.
- Checkpoints request model, optimizer, scheduler/RNG extra state, dataloader state, and direct HF weights.
- The queue order is quick, quick-resume, production-smoke, train, and launcher dry-run created no output root.
- Reward Parquet publication itself uses a same-directory temporary file and non-overwriting hard-link publication.

## Commands/checks run

- Fully read `AGENTS.md` and inspected the launcher, queue, runtime, reward, checkpoint, dataset, and exporter paths with `rg`, `nl`, and `sed`.
- Ran `python3 -m py_compile` on the focused training/checkpoint modules.
- Ran 8-GPU quick and train dry-runs and checked batch values and filesystem purity.
- Ran a CPU temporary checkpoint resolver fixture proving selection of partial step 201 over committed step 200.
- Repository pytest was unavailable because the local offline interpreter lacks pytest, Torch, PyArrow, Transformers, TorchData, and OmegaConf; no packages were installed.
