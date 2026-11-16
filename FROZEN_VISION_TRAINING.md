# Frozen Vision Encoder (FVE) training

## Portable runtime contract

FVE freezes the full `visual.*` module, including the visual merger, and trains
the remaining language-side parameters with GRPO on the Rotation task only.
Position, Jigsaw, and Contrastive are out of scope. The launcher requires the
following machine-specific inputs:

- `--model-path`: local model directory for `Qwen2.5-VL-3B-Instruct`;
- `--data-root`: directory containing
  `MMBench_RotationQA_90degree/{train.parquet,test.parquet}`;
- `--output-root`: durable location for all generated artifacts.

The default output root is:

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
```

The same values can be supplied as `FVE_MODEL_PATH`, `FVE_DATA_ROOT`, and
`FVE_OUTPUT_ROOT`. `FVE_PYTHON` optionally selects the Python interpreter.
No server-specific proxy, Hugging Face endpoint, or Python path is hard-coded.

For a run ID `fve-...`, durable output has this layout:

```text
<output-root>/
  cache/huggingface/
  smoke_data/
  state/<run-id>/{queue.complete,queue.failed}  # atomic JSON attempt state
  state/watcher.lock
  runs/<run-id>/<experiment>/
    checkpoints/
    tensorboard/
    reward_traces/<task>/step_XXXXXXXX.parquet
  runs/<run-id>/logs/
```

Ray temporary files use `/tmp/fve-ray` by default, configurable through
`FVE_RAY_TMP_ROOT` or `--ray-tmp-root`. This path is intentionally short:
Ray's UNIX-domain sockets have a 107-byte path limit. Ray temporary files are
not durable experiment outputs.

## First run on the 8 x H20 141 GiB machine

Build and validate the environment exactly as specified in `AGENTS.md`, then
download the pinned model and build the pinned Rotation dataset:

```bash
cd TPAMI-2027-zry

export FVE_OUTPUT_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
export FVE_MODEL_PATH="$FVE_OUTPUT_ROOT/inputs/models/Qwen2.5-VL-3B-Instruct"
export FVE_DATA_ROOT="$FVE_OUTPUT_ROOT/inputs/datasets"

mkdir -p "$FVE_OUTPUT_ROOT/inputs/models" "$FVE_DATA_ROOT"
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir "$FVE_MODEL_PATH"
python scripts/frozen_vision/prepare_rotation_data.py
python scripts/frozen_vision/prepare_rotation_data.py --verify-only

python scripts/frozen_vision/prepare_smoke_data.py
python scripts/frozen_vision/run_frozen_vision.py \
  --task rotation --mode quick --gpus 0,1,2,3,4,5,6,7 \
  --run-id h20-smoke-001
```

The smoke-data generator writes ten quick examples so that the quick and
quick-resume gates remain divisible on every supported GPU count, including 8.

After the smoke gates pass, start the full sequence:

```bash
python scripts/frozen_vision/run_queue.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --run-id h20-fve-001
```

The watcher defaults to waiting for all eight idle GPUs. It is optional on a
dedicated machine:

```bash
python scripts/frozen_vision/gpu_watcher.py --run-id h20-fve-001
```

## Batch invariants for 8 H20 GPUs

Production settings deliberately preserve the existing global batch:

| Quantity | Value |
|---|---:|
| Prompt batch | 512 |
| Rollouts per prompt | 5 |
| Global rollout responses | 2560 |
| Global PPO mini-batch | 640 |
| Per-H20 PPO mini-batch | 80 |
| Per-H20 micro-batch | 10 |
| Gradient accumulation per PPO mini-batch | 8 |

The 141 GiB H20 capacity does not by itself justify changing this batch: doing
so would change optimization semantics. The launcher therefore retains the
8-GPU exact-equivalence configuration and keeps vLLM memory utilization at
0.6 until an 8-GPU production-shape smoke test establishes actual headroom.

Production optimization reads only
`MMBench_RotationQA_90degree/train.parquet`; `test.parquet` is reserved for
validation and is never concatenated into `data.train_files`.
Before Ray starts, the launcher revalidates the pinned model revision and the
complete Rotation data manifest, schema, semantics, and file checksums.
The guarded queue also enforces the exact dependency family from `AGENTS.md`
and records it in `state/<run-id>/queue.complete`.

## Logs, reward traces, and checkpoints

View TensorBoard with:

```bash
tensorboard --logdir "$FVE_OUTPUT_ROOT/runs"
```

Tail a stage log with:

```bash
tail -F "$FVE_OUTPUT_ROOT/runs/<run-id>/logs/rotation_train.log"
```

After the queue completes, export or validate the latest HF checkpoint and run
the pinned MMBench smoke in one command:

```bash
python scripts/evaluation/run_post_training_smoke.py \
  --checkpoint "$FVE_OUTPUT_ROOT/runs/<run-id>/ssl4rl_qwen_3b_mmbench_rotation_frozen_vision/checkpoints" \
  --run-id <run-id> \
  --gpu <one-gpu-index>
```

The post-training manifest is published under the fingerprinted directory
`$FVE_OUTPUT_ROOT/evaluation/post_training/<run-id>/<fingerprint>` only after a
committed checkpoint exports and passes the MMBench smoke. Existing attempts
are never overwritten; `--resume` accepts only an identical completed attempt.
Its `export.exported_model` field is the authoritative HF model path for all
subsequent six-benchmark, MathVista, and MMMU evaluation commands. For a direct
HF checkpoint this normally ends in `global_step_<N>/actor/huggingface`; for a
shard merge it normally ends in `exports/global_step_<N>-hf`.

Every completed training step writes an atomic Parquet file containing one row
per rollout response. Each row records run ID, task, `global_step`, epoch, UID,
rollout index, dataset index, source, ground truth, generated response,
`raw_reward`, reward-computation UTC nanoseconds, and trace-write UTC
nanoseconds. Existing step files are never overwritten silently.

Hydra configuration snapshots and its job log are written under each
experiment's `hydra/` directory inside `FVE_OUTPUT_ROOT`; the training command
must not create a repository-local `outputs/` tree.

See `EVALUATION.md` and `AGENTS.md` for the six-benchmark run, Table I--III
aggregation, and final result-bundle commands.
