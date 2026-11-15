# Repository instructions for coding agents

## Scope and objective

This checkout contains the Rotation-only Frozen Vision Encoder (FVE) GRPO
workflow built on SSL4RL and the bundled `verl 0.4.1.dev` source. Position,
Jigsaw, and Contrastive are out of scope. The immediate deployment target is
one Linux job with 8 NVIDIA H20 GPUs (compute capability `sm_90`) and a durable
output root at:

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
```

Do not begin a full training run until the environment and all three FVE smoke
gates below pass. Do not change global batch semantics merely because the H20s
have more memory.

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
   python -c "import torch,vllm,tensordict,transformers,ray; print(torch.__version__,torch.version.cuda,torch.cuda.get_arch_list()); print(vllm.__version__,tensordict.__version__,transformers.__version__,ray.__version__); assert 'sm_90' in torch.cuda.get_arch_list()"
   ```

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
   python -m pytest -q tests/utils/test_reward_trace.py tests/special_sanity/test_import.py tests/utils/test_config_on_cpu.py
   ```

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

Choose exactly one GPU with sufficient free memory and run the base-model
MMBench v1.1 smoke. The wrapper downloads and MD5-validates the labeled
MMBench TSV when `--source-tsv` is omitted. It selects one complete circular
group for each of the six Table II subtasks and uses VLMEvalKit's official
Qwen2.5-VL adapter plus exact-matching evaluator.

```bash
CUDA_VISIBLE_DEVICES=<one-gpu-index> "$FVE_EVAL_ROOT/env/bin/python" \
  scripts/evaluation/vlmeval_mmbench_smoke.py \
  --model-path "$FVE_MODEL_PATH" \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval" \
  --output-dir "$FVE_OUTPUT_ROOT/evaluation/smoke/mmbench_base_qwen25vl3b"
```

Require all of the following before starting the long RL queue:

- 24 predictions covering Logical, Relation, Attribute, Coarse, Cross-Inst.,
  and Single-Inst.;
- VLMEvalKit prediction, exact-matching result, and accuracy files;
- `mmbench_v11_smoke_table2.json` with all six subtasks plus Average;
- `smoke_manifest.json` recording model, GPU, dataset hash, sample indices,
  Torch/Transformers versions, and VLMEvalKit commit.

Smoke accuracy is not a reportable experiment result. Its purpose is to prove
that the exact evaluation implementation can load the model, process images,
write predictions, score complete circular groups, and project the official
subtasks into the paper's Table II columns.

### 8. Run the guarded Rotation queue

Use a new explicit run ID. The queue must pass Rotation `quick`,
`quick-resume`, and `production-smoke`, in order, before the single Rotation
training stage starts.

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
$FVE_OUTPUT_ROOT/evaluation/post_training/<training-run-id>/
  export_result.json
  mmbench_smoke.log
  mmbench_smoke/
  post_training_manifest.json
```

Do not start full six-benchmark evaluation unless this post-training manifest
exists and the exported model passes the same six-subtask smoke as the base
model.

### 10. Run six benchmarks, aggregate the paper tables, and package results

Download and checksum all six paper datasets. The paper protocol is
MMBench DEV-EN v1.1, the nine image dimensions of SEEDBench_IMG, VStarBench,
RealWorldQA, BLINK, and MME-RealWorld-Lite (the paper's abbreviated `MME`
column is not standard MME).

```bash
python scripts/evaluation/prepare_evaluation_data.py \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval"
python scripts/evaluation/prepare_evaluation_data.py \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval" \
  --verify-only
```

Evaluate the base and exported Rotation models with identical paper sampling
parameters (`temperature=0.01`, `top-p=0.001`, `top-k=1`):

```bash
python scripts/evaluation/run_six_benchmarks.py \
  --model-path "$FVE_MODEL_PATH" --model-name Base \
  --run-id <training-run-id> --gpu <one-gpu-index> \
  --output-root "$FVE_OUTPUT_ROOT"

python scripts/evaluation/run_six_benchmarks.py \
  --model-path <exported-hf-model> --model-name Rotation \
  --run-id <training-run-id> --gpu <one-gpu-index> \
  --output-root "$FVE_OUTPUT_ROOT"
```

Generate Table I plus the MMBench and SEED-Bench detail tables:

```bash
python scripts/evaluation/aggregate_paper_tables.py \
  --base-manifest "$FVE_OUTPUT_ROOT/evaluation/full/<training-run-id>/Base/six_benchmark_run_manifest.json" \
  --rotation-manifest "$FVE_OUTPUT_ROOT/evaluation/full/<training-run-id>/Rotation/six_benchmark_run_manifest.json" \
  --output-dir "$FVE_OUTPUT_ROOT/evaluation/tables/<training-run-id>"
```

The screenshot containing Golden-3B uses separate 60/40 held-out MMBench and
SEED-Bench detail runs; those detail averages are not identical to Table I.
When reproducing that protocol, pass the four explicit `--*-detail` score-file
overrides supported by `aggregate_paper_tables.py`. Never substitute the
Table I aggregate for a held-out detail score.

Finally create a checksum sidecar and compressed, curated result bundle. Large
checkpoints, datasets, and per-step reward Parquets are inventoried rather than
duplicated; tables, predictions, score artifacts, logs, and manifests are
included.

```bash
python scripts/evaluation/package_final_results.py \
  --run-id <training-run-id> \
  --training-run-dir "$FVE_OUTPUT_ROOT/runs/<training-run-id>" \
  --base-eval-manifest "$FVE_OUTPUT_ROOT/evaluation/full/<training-run-id>/Base/six_benchmark_run_manifest.json" \
  --rotation-eval-manifest "$FVE_OUTPUT_ROOT/evaluation/full/<training-run-id>/Rotation/six_benchmark_run_manifest.json" \
  --tables-dir "$FVE_OUTPUT_ROOT/evaluation/tables/<training-run-id>" \
  --output-dir "$FVE_OUTPUT_ROOT/final"
```

## Editing and validation rules

- Keep generated models, datasets, caches, logs, traces, and checkpoints out of
  Git.
- Do not restore old `/data2/ryzhou/...` paths or machine-specific proxies.
- Preserve dry-run purity: `--dry-run` must not create directories or files.
- Use `python3` or the selected environment's explicit Python, never assume a
  `python` alias outside the image.
- After changing FVE launch code, run `python -m py_compile` on all files under
  `scripts/frozen_vision`, run the lightweight tests above, and dry-run the
  8-GPU quick and train modes.
- Never start full training merely because imports pass. Real CUDA,
  FlashAttention, quick-resume, and production-shape tests are mandatory.
