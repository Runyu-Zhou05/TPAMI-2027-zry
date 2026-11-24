# Repository instructions for coding agents

## Scope and objective

This checkout contains the Rotation-only Frozen Vision Encoder (FVE) GRPO
workflow built on SSL4RL and the bundled `verl 0.4.1.dev` source. Position,
Jigsaw, and Contrastive remain out of scope for the FVE queue. A separate
reviewer-analysis extension may train the four original unfrozen SSL4RL tasks
and evaluate their exported checkpoints on MathVista/MMMU; it must not reuse
the FVE queue or label normal checkpoints as FVE exports. The immediate
deployment target is one Linux job with 8 NVIDIA H20 GPUs (compute capability
`sm_90`) and a durable output root at:

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
```

Do not begin a full training run until the environment and all three FVE smoke
gates below pass. Do not change global batch semantics merely because the H20s
have more memory.

Request at least 180 GB pod memory (about 167 GiB). FVE enables parameter and
optimizer CPU offload; the previously attempted 64 GB limit OOMed before a
training step. The environment gate fails below 160 GiB effective cgroup memory.

## Environment source of truth

Use a Python 3.10 image and prefer the following validated core stack:

| Package | Version |
|---|---:|
| Python | 3.10 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| vLLM | 0.10.2 |
| FlashAttention | 2.8.3.post2 |
| xFormers | 0.0.32.post1 |
| Triton | 3.4.0 |
| TensorDict | 0.9.1 |
| torchdata | 0.11.0 |
| Transformers | 4.55.2 |
| tokenizers | 0.21.1 |
| Accelerate | 1.8.1 |
| PEFT | 0.21.2 |
| Ray | 2.60.0 |
| datasets | 4.0.0 |
| PyArrow | 20.0.0 |
| qwen-vl-utils | 0.0.11 |
| NumPy | 1.26.4 |
| Hydra Core | 1.3.8 |
| OmegaConf | 2.3.1 |

This exact family was checked on 2026-11-15 in the existing SSL4RL environment
on `ryzhou@222.29.2.247`: `pip check` was clean; six CPU repository tests
passed; FVE, FSDP, reward-trace, and vLLM integration modules imported together;
and real CUDA plus FlashAttention kernels ran successfully. The PyTorch cu128
wheel reports both `sm_90` (H20) and `sm_120` support. This is strong package
compatibility evidence, but it is not a substitute for the required H20 smoke
run: no complete FVE optimizer step has yet succeeded on the target machine.

The host NVIDIA driver must support CUDA 12.8 containers. If it does not, stop
and report the driver and image constraints. Do not silently substitute an
older package family.

## Known dependency conflicts: do not mix these stacks

The generic upstream documentation and installer predate this FVE handoff:

- `README.md` describes `verl 0.3.0`, while this checkout bundles
  `verl 0.4.1.dev`.
- `scripts/install_vllm_sglang_mcore.sh` pins Torch 2.6.0, vLLM 0.8.5.post1,
  TensorDict 0.6.2, FlashAttention 2.7.4, and CUDA 12.4-era wheels.
- `setup.py` requires TensorDict 0.8.0 through 0.9.1, so the old install
  script's TensorDict 0.6.2 conflicts with the checked-in code.
- The optional `setup.py` extra named `vllm` caps vLLM at 0.8.5 and therefore
  does not describe the validated vLLM 0.10.2 stack.

Consequently:

1. Do **not** run `scripts/install_vllm_sglang_mcore.sh`.
2. Do **not** install `.[vllm]` or `.[sglang]`.
3. Install the validated Torch/vLLM/core versions first, then install this
   repository with plain `pip install -e .`.
4. Finish with `pip check` and the gates below. A successful import alone is
   insufficient evidence of CUDA compatibility.

## Suggested image-build order

Adapt package-index and wheel locations to the image service, but preserve the
versions and order unless a failed gate provides concrete evidence to change
them:

```bash
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm==0.10.2 xformers==0.0.32.post1 tensordict==0.9.1 torchdata==0.11.0
python -m pip install transformers==4.55.2 tokenizers==0.21.1 accelerate==1.8.1 peft==0.21.2 datasets==4.0.0 pyarrow==20.0.0 qwen-vl-utils==0.0.11 numpy==1.26.4 hydra-core==1.3.8 omegaconf==2.3.1 ray[default]==2.60.0 pytest
MAX_JOBS=8 python -m pip install flash-attn==2.8.3.post2 --no-build-isolation
python -m pip install -e .
python -m pip check
```

Do not add a separate FlashInfer pin unless vLLM reports that it is required.
The validated vLLM 0.10.2 environment had no standalone FlashInfer package;
the separately installed FlashAttention package passed a real CUDA kernel test.

## Mandatory environment gates

Run these checks inside the final image, with the selected environment first on
`PATH`. Preserve their output in the job log.

1. Confirm versions and H20 kernel coverage:

   ```bash
   python scripts/frozen_vision/check_environment.py
   ```

   This also executes the Qwen2.5-VL `apply_monkey_patch` path against the
   installed Transformers version and checks effective pod/cgroup memory. An
   import-only check is insufficient.

2. Run a real CUDA operation on every visible GPU, not just an import:

   ```bash
   python -c "import torch; assert torch.cuda.device_count()==8; print([(i,torch.cuda.get_device_name(i),(torch.ones(1,device=f'cuda:{i}')+1).item()) for i in range(8)])"
   ```

3. Run a real FlashAttention kernel:

   ```bash
   python -c "import torch; from flash_attn import flash_attn_func; q=torch.randn(1,4,2,64,device='cuda',dtype=torch.float16); o=flash_attn_func(q,q,q); torch.cuda.synchronize(); print(o.shape,torch.isfinite(o).all().item())"
   ```

4. Run the lightweight repository tests:

   ```bash
   python -m pytest -q \
     tests/workflow/test_source_tree_contracts.py \
     tests/utils/test_reward_trace.py \
     tests/special_sanity/test_import.py \
     tests/utils/test_config_on_cpu.py
   ```

   `test_import.py` must import `verl.workers.fsdp_workers`; a top-level
   `import verl` does not prove that the required `verl/models/` source package
   was included in the clone.

5. Confirm the Ray CLI belongs to the same environment. Use the executable,
   not `python -m ray` (Ray has no `ray.__main__`):

   ```bash
   "$(dirname "$(command -v python)")/ray" --version
   "$(dirname "$(command -v python)")/ray" stop --help >/dev/null
   ```

If any gate fails, record the complete command, versions, and error before
changing packages. Change one compatibility family at a time and repeat all
gates.

## FVE runtime invariants

- Base model: Qwen2.5-VL-3B-Instruct.
- The only supported SSL4RL task is Rotation, backed by
  `MMBench_RotationQA_90degree/{train.parquet,test.parquet}`.
- Optimize on `train.parquet` only. Reserve `test.parquet` exclusively for
  validation; never concatenate it into `data.train_files`.
- Freeze the complete `visual.*` module, including the merger.
- Keep `actor.fsdp_config.use_orig_params=True`.
- Expected parameters with the real checkpoint: 668,684,288 frozen visual;
  3,085,938,688 trainable; 3,754,622,976 total.
- For 8 GPUs, retain 512 prompts, 5 rollouts per prompt, global rollout batch
  2560, global PPO mini-batch 640, per-GPU PPO mini-batch 80, and micro-batch 10.
- Use a short local Ray temp root such as `/tmp/fve-ray`; Ray UNIX sockets have
  a 107-byte path limit.
- Write durable checkpoints, TensorBoard events, reward traces, state markers,
  and queue logs below `FVE_OUTPUT_ROOT`.

## Required execution order on the target job

Follow this order exactly. Do not start input downloads before the image is
built and the environment gates above pass.

### 1. Build and validate the environment

Build the image with the validated dependency family, run `pip check`, and run
all five mandatory environment gates above. Keep the gate output in the job
log. Stop here if any gate fails.

### 2. Define durable input and output paths

```bash
export FVE_OUTPUT_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
export FVE_MODEL_PATH="$FVE_OUTPUT_ROOT/inputs/models/Qwen2.5-VL-3B-Instruct"
export FVE_DATA_ROOT="$FVE_OUTPUT_ROOT/inputs/datasets"
export FVE_PYTHON="$(command -v python)"
mkdir -p "$FVE_OUTPUT_ROOT/inputs/models" "$FVE_DATA_ROOT"
```

### 3. Download the pinned base model

Download the exact Qwen revision used by this project. Do not use an unpinned
`main` revision.

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir "$FVE_MODEL_PATH"
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('$FVE_MODEL_PATH'); print(c.model_type,c.architectures); assert c.model_type=='qwen2_5_vl'"
```

The expected checkpoint has two safetensors shards with total tensor bytes
`7,509,245,952`. The real FVE worker performs the authoritative parameter-count
and frozen-module audit when the smoke run starts.

### 4. Download pinned raw MMBench and generate Rotation data

Run the project-owned data command. It downloads only the raw MMBench README
and Parquet files at revision
`215b79e126bea0777a489683f239edd0a6012a8a`, maps raw `test` to training and raw
`validation` to testing, deterministically generates 90-degree Rotation pairs
with seed 42, and writes the final RL Parquet schema.

```bash
python scripts/frozen_vision/prepare_rotation_data.py
python scripts/frozen_vision/prepare_rotation_data.py --verify-only
```

The final files must be:

```text
$FVE_DATA_ROOT/MMBench_RotationQA_90degree/train.parquet  # 6,666 rows
$FVE_DATA_ROOT/MMBench_RotationQA_90degree/test.parquet   # 4,329 rows
$FVE_DATA_ROOT/MMBench_RotationQA_90degree/manifest.json
```

The manifest records the source revision, raw-file hashes, generated-file
hashes, schema, row counts, package versions, seed, split mapping, and generator
Git commit. A rerun with a matching manifest validates and reuses the existing
data. The script refuses to trust pre-existing Parquet files without a matching
manifest. Do not use `--max-rows-per-split` for training; it is only a bounded
development test.

### 5. Generate Rotation smoke data

```bash
python scripts/frozen_vision/prepare_smoke_data.py
```

This reads only the generated Rotation training Parquet and writes the quick
and production-shape subsets below `FVE_OUTPUT_ROOT/smoke_data`.

### 6. Dry-run the 8-GPU production configuration

Inspect the printed input paths, output paths, and batch values. The global
rollout batch must be 2560 and the global PPO mini-batch must be 640.

```bash
python scripts/frozen_vision/run_frozen_vision.py --task rotation --mode train --gpus 0,1,2,3,4,5,6,7 --run-id h20-config-check --dry-run
```

### 7. Validate the post-training evaluation chain before training

Install the pinned VLMEvalKit revision in a separate overlay environment. This
must not modify the validated FVE training environment. See `EVALUATION.md`
for the validated 247 smoke evidence and the six-benchmark evaluation scope.

```bash
export FVE_EVAL_ROOT="$FVE_OUTPUT_ROOT/tools/vlmevalkit"
bash scripts/evaluation/setup_vlmevalkit.sh "$FVE_EVAL_ROOT"
```

This step only installs and validates the evaluator; do not launch full
inference yet. Step 10 later runs one intentional Base calibration on old
MMBench DEV-EN, followed by the exported Rotation checkpoint on all six
benchmarks.

### 8. Run the guarded Rotation queue

Use a new explicit run ID. The queue must pass Rotation `quick`,
`quick-resume`, and `production-smoke`, in order, before the single Rotation
training stage starts. It reruns the exact dependency-version/H20-kernel gate
and records the validated environment in its atomic completion manifest.

```bash
python scripts/frozen_vision/run_queue.py --gpus 0,1,2,3,4,5,6,7 --run-id <unique-run-id>
```

Do not bypass a failed smoke gate. Inspect the corresponding log and
`state/<run-id>/queue.failed` first. Do not claim H20 readiness until the
production-shape smoke completes and the log contains the frozen-vision runtime
audit confirming no visual gradients and an unchanged visual checksum.

### 9. Export the final checkpoint and repeat the MMBench smoke

FVE checkpoints save both resumable FSDP shards and standard HF weights. The
exporter validates direct HF weights when present and falls back to merging
older shard-only checkpoints. It rejects exports unless the model has exactly
3,754,622,976 total parameters and 668,684,288 visual parameters.

```bash
python scripts/evaluation/run_post_training_smoke.py \
  --checkpoint "$FVE_OUTPUT_ROOT/runs/<training-run-id>/ssl4rl_qwen_3b_mmbench_rotation_frozen_vision/checkpoints" \
  --run-id <training-run-id> \
  --gpu <one-gpu-index>
```

This resolves the latest `global_step_*`, exports or validates its HF model,
reuses the pinned VLMEvalKit environment, runs the same 24-sample MMBench
smoke, and writes:

```text
$FVE_OUTPUT_ROOT/evaluation/post_training/<training-run-id>/<attempt-fingerprint>/
  export_result.json
  mmbench_smoke.log
  mmbench_smoke/
  post_training_manifest.json
```

Do not start full six-benchmark evaluation unless this post-training manifest
exists and the exported Rotation model passes its six-subtask smoke.

Before evaluation, require the train-stage reward difficulty analysis manifest
under the run directory. New queues generate it automatically. For a run
started from an older checkout, run `analyze_reward_difficulty.py` manually as
documented in `REWARD_DIFFICULTY.md`. The primary reviewer-facing metric is the
fraction of prompt groups with non-zero within-group reward variance; do not
substitute global reward standard deviation.
Legacy queues do not contain this manifest in `queue.complete`; after manual
analysis, pass its path to `package_final_results.py` with
`--reward-difficulty-manifest`.

### 10. Reproduce the paper evaluation protocol

Download and checksum all six paper datasets. The reproduction protocol is old
MMBench DEV-EN, the nine image dimensions of SEEDBench_IMG, VStarBench,
RealWorldQA, BLINK, and MME-RealWorld-Lite (the paper's abbreviated `MME`
column is not standard MME).

```bash
python scripts/evaluation/prepare_evaluation_data.py \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval" \
  --benchmarks MMBench_DEV_EN SEEDBench_IMG VStarBench RealWorldQA BLINK MME-RealWorld-Lite
python scripts/evaluation/prepare_evaluation_data.py \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval" \
  --benchmarks MMBench_DEV_EN SEEDBench_IMG VStarBench RealWorldQA BLINK MME-RealWorld-Lite \
  --verify-only
```

Run exactly two fresh, fingerprinted attempts: Base on old MMBench DEV-EN to
test reproduction of the paper's 72.99 baseline, and exported Rotation on all
six benchmarks. Both use `min_pixels=35840`, the verified generic ImageMCQ COT
prompt, and paper sampling (`temperature=0.01`, `top-p=0.001`, `top-k=1`). The
pinned `MMERealWorld` class intentionally retains its own answer-only prompt,
which matches the upstream evaluator path rather than inventing a new patch.
Do not use `--resume` on the first invocation and do not copy/reuse old
direct-answer predictions.

```bash
python scripts/evaluation/run_six_benchmarks.py \
  --model-path "$FVE_MODEL_PATH" --model-name BasePaper \
  --model-role base --protocol paper-readme --benchmark-scope mmbench \
  --run-id <training-run-id>-base-paper --gpu <one-gpu-index> \
  --output-root "$FVE_OUTPUT_ROOT"

python scripts/evaluation/run_six_benchmarks.py \
  --model-path <exported-hf-model> --model-name Rotation \
  --model-role rotation --protocol paper-readme --benchmark-scope six \
  --run-id <training-run-id>-rotation-paper --gpu <one-gpu-index> \
  --output-root "$FVE_OUTPUT_ROOT"
```

The runner permits only `Base+MMBench` and `Rotation+six` under
`paper-readme`. Keep both manifests, configs, logs, predictions, score files,
and the reported reasoning-format coverage. Base must be compared to 72.99
before interpreting the Rotation-to-paper comparison.

### 11. Run MathVista and MMMU chain-of-thought evaluation

Use `MathVista_MINI` and `MMMU_DEV_VAL`; do not silently substitute unlabeled
test splits. The legacy FVE path evaluates its exported Rotation model. The
reviewer-analysis path evaluates all four normal paper-recipe tasks: Position,
Rotation, Jigsaw, and Contrastive. Use the explicit
`<think>...</think>` and `<answer>...</answer>` reasoning prompts, 4,096
response tokens, and deterministic generation settings. Raw reasoning remains
in prediction workbooks. MathVista uses the official DeepSeek `deepseek-chat`
API as its auxiliary extraction judge. For MMMU, the runner's built-in exact
matching and strict `<answer>` extraction are diagnostics only: exact matching
cannot reliably parse long tagged CoT, while tag-only scoring confounds answer
accuracy with format compliance. The reportable MMMU score must come from a
separate, model-blind `deepseek-chat` answer-extraction pass over the complete
saved predictions. The
trusted handoff stores the credential only as Base64 in
`deepseek_api_key_b64.py`; the runner decodes it in memory and must never print
or place the plaintext in logs or manifests.

MMMU DeepSeek rescoring never reruns GPU inference. For every one of the 1,050
`MMMU_DEV_VAL` rows (900 validation and 150 development), send exactly the
question, options, and candidate response to the same fixed extraction prompt;
never send the ground-truth answer or model identity. Use the official DeepSeek
endpoint, `deepseek-chat`, temperature 0, and require one `A`--`D` letter or an
explicit invalid result. Retry transient API failures, but never drop a row or
count an unresolved request silently; publication requires 1,050 completed
records, zero API errors, and zero NaNs. Apply the extracted letters to ground
truth only after the judge response has been saved.

Publish an immutable per-model rescore directory containing the extraction
prompt and SHA256, source prediction workbook SHA256, one auditable row per
dataset index (split, prediction hash, extracted letter/status, correctness),
validation/development/overall scores, parsing coverage, API-error inventory,
and a manifest recording endpoint/model/decoding settings without the key.
Base and trained models must use byte-identical judge settings. Preserve the
legacy exact/tag scores as diagnostics, and use paired per-index outcomes for
McNemar or paired-bootstrap comparisons. Do not let
`aggregate_reasoning_results.py` present legacy MMMU exact matching as the
primary result; until it consumes the DeepSeek-rescore manifest, publish the
rescored MMMU table separately.

Before GPU evaluation, verify the bundled credential and both judge paths
without loading a model:

```bash
python scripts/evaluation/smoke_reasoning_judges.py
```

```bash
python scripts/evaluation/prepare_reasoning_data.py \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval"

python scripts/evaluation/run_reasoning_benchmarks.py \
  --model-path <export.exported_model> --model-name Rotation --model-role rotation \
  --run-id <training-run-id> --gpu <one-gpu-index> \
  --output-root "$FVE_OUTPUT_ROOT" --mathvista-judge deepseek-chat

python scripts/evaluation/aggregate_reasoning_results.py \
  --rotation-manifest <rotation-reasoning-manifest> \
  --output-dir "$FVE_OUTPUT_ROOT/evaluation/reasoning_tables/<training-run-id>"
```

For each normal paper-recipe task, export the committed checkpoint with its
true task identity, then run the same reasoning command with the matching role:

```bash
python scripts/evaluation/export_fve_checkpoint.py \
  --checkpoint <task-checkpoint-root> --artifact-type ssl4rl_export \
  --model-role <position|rotation|jigsaw|contrastive> \
  --result-json <task-export-result.json>

python scripts/evaluation/run_reasoning_benchmarks.py \
  --model-path <exported_model> --model-name <display-name> \
  --model-role <position|rotation|jigsaw|contrastive> \
  --run-id <task-training-run-id> --gpu <one-gpu-index> \
  --output-root "$FVE_OUTPUT_ROOT" --mathvista-judge deepseek-chat
```

After all four complete, aggregate only compatible attempts:

```bash
python scripts/evaluation/aggregate_reasoning_results.py \
  --position-manifest <position-manifest> \
  --rotation-manifest <rotation-manifest> \
  --jigsaw-manifest <jigsaw-manifest> \
  --contrastive-manifest <contrastive-manifest> \
  --output-dir "$FVE_OUTPUT_ROOT/evaluation/reasoning_tables/four-task"
```

The legacy Rotation-only aggregation remains supported. See `EVALUATION.md`
for DeepSeek routing, credential handling, provenance, and output details.

### 12. Preserve result artifacts

Preserve the queue, reward analysis, post-training, BasePaper MMBench, full
Rotation, detail Rotation, and reasoning manifests and their fingerprinted
attempt directories. Do not substitute the BasePaper MMBench-only manifest for
a legacy six-benchmark Base manifest: the paper-reproduction attempts are a
separate comparison and must remain immutable.

### 13. Run the reviewer visual-feature probe

Use the fixed ImageNet-100 protocol in `PROBING.md` for Base plus all four
normal unfrozen checkpoints: Rotation, Position, Jigsaw, and Contrastive.
Extract both final-ViT/pre-merger and post-merger/pre-LLM features at a fixed
224 crop, mean pool all spatial tokens, and train only the linear heads. Every
normal task checkpoint must carry a matching `ssl4rl_export_manifest.json`;
never relabel an FVE export.

The canonical dataset identity is the 100-class ImageFolder tree documented in
`PROBING.md` (130,000 train and 5,000 validation images), not
`PKU-ML/ImageNet-Subset`. Run `smoke` before full extraction. Server 247 may be
used only for the bounded Base smoke because `/data1` and `/data2` are nearly
full; store full feature caches under the target H20 job's durable output root.

### 14. Run Position-to-Golden RLVR

Follow `GOLDEN_RLVR.md`. Use the checksum-pinned old `MMBench_DEV_EN`, its
deterministic seed-42 complete-group 60/40 split, and the final normal Position
HF export. Golden training is a fresh unfrozen run, never a checkpoint resume:
20 epochs, an explicit 100-step ceiling, validation before training, and
validation/checkpoints every 20 steps. Preserve all checkpoints, reward traces,
and validation JSONL files. The launcher fails closed unless terminal step 100
and validation steps 0/20/40/60/80/100 are present.

For the separate SEED-Bench replication, follow `SEED_GOLDEN_RLVR.md`. It must
start from the final normal **SEED-Position** checkpoint, never the MMBench
Position or FVE checkpoint. The checksum-pinned 14,232-row `SEEDBench_IMG` is
split by its nine categories into 8,539 train and 5,693 held-out rows. The
fixed budget is 20 epochs / 320 steps with validation before training and every
20 steps; the launcher requires committed step 320 plus all validation dumps.

### 15. Run the four Table-XI harder variants

Follow `DIFFICULTY_VARIANTS.md` for Rotation45, Jigsaw3x3,
Contrastive-strong, and Position3x3. Generate each hard dataset into its own
directory, convert it with `prepare_hard_variant_data.py`, and pass the
one-step real smoke before full training. These are normal unfrozen 3B runs,
not FVE runs. Preserve the original paper launcher semantics, including the
train+test optimization inputs, while recording per-rollout reward traces.
Jigsaw3x3 must explicitly allow nine images per prompt.

## Editing and validation rules

- Keep generated models, datasets, caches, logs, traces, and checkpoints out of
  Git.
- Keep the generated root directory ignore anchored as `/models/`. Never change
  it back to an unanchored `models/`, which would silently hide the required
  `verl/models/` Python package.
- Do not restore old `/data2/ryzhou/...` paths or machine-specific proxies.
- Preserve dry-run purity: `--dry-run` must not create directories or files.
- Use `python3` or the selected environment's explicit Python, never assume a
  `python` alias outside the image.
- After changing FVE launch code, run `python -m py_compile` on all files under
  `scripts/frozen_vision`, run the lightweight tests above, and dry-run the
  8-GPU quick and train modes.
- Never start full training merely because imports pass. Real CUDA,
  FlashAttention, quick-resume, and production-shape tests are mandatory.
