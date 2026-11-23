# FVE post-training evaluation

## Target execution policy

The paper-reproduction job has exactly two allowed evaluations: the untouched
pinned `Qwen/Qwen2.5-VL-3B-Instruct` Base on old `MMBench_DEV_EN`, and the
exported Rotation checkpoint on all six benchmarks. Both use the literal README
pixel budget and the verified generic ImageMCQ COT prompt. MathVista/MMMU remain
Rotation-only. Other Base/full-benchmark combinations are rejected.

## Evaluation objective

After Rotation FVE training, evaluate the exported model on the original VQA
tasks rather than on Rotation-generated questions:

- MMBench;
- SEED-Bench;
- VStarBench (V*), the standard MCQ benchmark used by the paper to measure
  difficult visual-search scenarios;
- RealWorldQA;
- BLINK;
- MME.

MMBench must report Logical, Relation, Attribute, Coarse, Cross-Instance,
Single-Instance, and Average. SEED-Bench must report the nine subtasks shown in
the paper table plus Average. Table I uses the six benchmark-level scores and
their unweighted average.

The authoritative paper protocol uses MMBench DEV-EN, the nine image dimensions
of SEED-Bench (14,232 examples), VStarBench, RealWorldQA, BLINK, and
MME-RealWorld-Lite. Evaluation uses VLMEvalKit with temperature 0.01, top-p
0.001, and top-k 1. V* is evaluated as the paper's standard VStarBench MCQ
dataset; no additional crop/tool agent loop is introduced.

The existing `build_benchmarks/makerotation_*.py` files construct SSL4RL
Rotation tasks from benchmark images. They are not the post-training benchmark
evaluation implementation. `makerotation_mme.py` uses MME-RealWorld-Lite,
which is also the benchmark abbreviated as `MME` in the paper table.

## Pinned VLMEvalKit

Use:

```text
repository: https://github.com/open-compass/VLMEvalKit.git
commit:     0a11917eede83742389b03d7f11fc252aabdf6ef
date:       2025-03-04
```

The earlier `v0.1` tag predates Qwen2.5-VL. Commit `2d7808e1...` first added the
Qwen2.5-VL-3B registration but contained a same-day `SmolVLM2` import bug;
`0a11917e...` is the immediate fix. Keep VLMEvalKit in the isolated environment
created by `scripts/evaluation/setup_vlmevalkit.sh` so its dependencies cannot
change the FVE training environment.

The setup script applies one scoped compatibility patch to this pinned commit:
the historical Qwen adapter guessed Qwen2.5 from the directory name. The patch
uses `config.json` model type and architectures instead, so exported checkpoint
directories can use arbitrary run and step names.

It also applies the SSL4RL upstream README's MCQ prompt patch, requiring a
`<think>...</think> <answer>$LETTER</answer>` final response. The normal
`current` protocol uses the corrected Qwen/VLMEvalKit value
`1280×28×28 = 1,003,520`. The dedicated `paper-readme` reproduction protocol
instead preserves the historical literal value `min_pixels=35840` so the Base
run can test whether that setting explains the reported 72.99. Both protocols
use `max_pixels=16384×28×28 = 12,845,056`.

The setup gate verifies the exact tag sequence in the generic `image_mcq.py`
prompt. Do not use the generic phrase "The last line..." as the patch marker:
the pinned source already contains it in an unrelated MMMU-Pro branch, which
previously caused a false "already applied" result and answer-only evaluation.

## Validated MMBench smoke on 247

On 2026-11-15, the complete base-model smoke succeeded with:

```text
model:       Qwen2.5-VL-3B-Instruct
dataset:     MMBench_DEV_EN_V11
model stack: Torch 2.8.0+cu128, Transformers 4.55.2
GPU:         physical GPU 9, NVIDIA RTX PRO 5000 72GB Blackwell
samples:     24, one complete four-way circular group per Table II subtask
evaluator:   VLMEvalKit exact_matching
```

The durable remote artifacts are under:

```text
/data2/ryzhou/TPAMI_rebuttal/evaluation/smoke/mmbench_base_qwen25vl3b/
```

They include the selected TSV, prediction workbook, exact-matching detail,
official accuracy CSV, Table II projection JSON, log, and manifest. All six
subtask fields and Average were produced. The 24-sample score happened to be
100%; this is only a chain smoke and must not be reported as benchmark
performance.

No processes or GPU memory from the smoke remained after exit, and the original
FVE training environment remained dependency-clean.

## Validated FSDP-to-HF export on 247

The post-training export path was tested with a full Qwen2.5-VL-3B state dict
stored in the same single-rank FSDP file layout used by verl. The fallback
merger produced two HF safetensors shards and passed these exact audits:

```text
model_type:    qwen2_5_vl
total params:  3,754,622,976
visual params:   668,684,288
tensor count:  824
```

Testing exposed and fixed two compatibility bugs in the bundled verl export
code:

1. Transformers 4.55 `AutoModel.from_config` no longer accepts
   `torch_dtype`; the module is now meta-initialized and explicitly cast.
2. FSDP merging materialized tied input embeddings and `lm_head` as separate
   tensors; the exporter now verifies equality and omits the duplicate head
   before serialization.

The exported model then produced predictions identical to the original base
model on all 24 MMBench smoke rows and passed the official exact-matching
evaluator. Large temporary checkpoint and exported-weight fixtures were
removed after validation; small manifests, logs, and score evidence remain at:

```text
/data2/ryzhou/TPAMI_rebuttal/evaluation/smoke/export_fixture_evidence/
```

The normal direct-HF checkpoint path was also validated separately with
hardlinked base weights: the exporter selected `direct_hf_checkpoint`, repeated
the exact parameter audit, wrote its manifest, and the fixture was removed
without duplicating or altering the base model.

## Required evaluation chain

The remaining implementation should reuse the validated adapter and result
schema:

```text
FVE FSDP checkpoint
  -> portable Hugging Face export
  -> identical decoding configuration for base and FVE models
  -> VLMEvalKit evaluation for MMBench, SEED-Bench, RealWorldQA, BLINK, MME
  -> VLMEvalKit VStarBench evaluation
  -> per-sample predictions and official detail files
  -> Table I, Table II, and Table III aggregation
```

Before any full benchmark run, repeat the same MMBench smoke with an exported
quick FVE checkpoint. This is the acceptance test for the checkpoint exporter.
Use `scripts/evaluation/run_post_training_smoke.py` to perform checkpoint
resolution, export validation, and MMBench smoke as one guarded command.

## Tables and final package

`prepare_evaluation_data.py` pins URLs and checksums for all six datasets.
`run_six_benchmarks.py` builds the fixed VLMEvalKit config and refuses success
unless every benchmark emits an official score artifact.

For literal paper reproduction, the only valid jobs are:

```text
BasePaper: model-role=base, protocol=paper-readme, benchmark-scope=mmbench
Rotation:  model-role=rotation, protocol=paper-readme, benchmark-scope=six
```

Both use old `MMBench_DEV_EN`, `min_pixels=35840`, `max_pixels=12845056`,
`max_new_tokens=2048`, and the exact verified SSL4RL COT prompt. The fingerprint
and output directory differ from all earlier direct-answer attempts, and the
first invocation must not use `--resume`. This generic prompt covers MMBench,
SEEDBench_IMG, VStarBench, RealWorldQA, and BLINK. The pinned
`MMERealWorld` subclass overrides `build_prompt` with its upstream answer-only
instruction; leaving that override intact is part of reproducing the actual
evaluator rather than silently changing a seventh configuration variable.

`prepare_detail_splits.py` implements the paper's downstream-benchmark 60/40
protocol for Tables II/III. It preserves complete MMBench circular groups,
stratifies SEED rows by category, fixes seed 42 (the paper does not publish a
seed), and records exact train/held-out indices, source hashes, subset hashes,
and the split ID. `run_detail_benchmarks.py` binds each Base/Rotation detail
attempt to that split and to the same model/evaluator/decoding fingerprints as
its full run. Held-out TSVs use explicit `_FVE_HELDOUT` custom dataset names;
reusing official dataset names would trigger VLMEvalKit's full-set MD5/download
path and invalidate the split.

`aggregate_paper_tables.py` writes CSV, JSON, and LaTeX for:

- Table I: six benchmark scores and their unweighted average;
- Table II: MMBench Logical, Relation, Attribute, Coarse, Cross-Instance,
  Single-Instance, and weighted Overall;
- Table III: the nine SEED-Bench image dimensions and weighted Overall.

Table II/III inputs are paired detail-run manifests, never raw score files. The
aggregator requires either no detail manifests or a compatible Base/Rotation
pair with the same run ID, exact split, models, evaluator, and generation
protocol. This is required for the paper screenshot's Golden-3B 60/40 held-out
comparison, whose detail averages differ from Table I's full-benchmark scores.

`package_final_results.py` creates a curated `tar.gz`, an SHA256 sidecar, and an
artifact manifest only after validating the complete queue/data/checkpoint/
post-training/evaluation/table lineage. It includes tables, predictions,
official score files, logs, and manifests while stream-hashing and inventorying
rather than duplicating large datasets, checkpoints, TensorBoard files, and
per-step reward Parquets.

## MathVista and MMMU reasoning evaluation

The additional reasoning chain is deliberately separate from Tables I--III. It
supports both the legacy exported Rotation FVE model and the four normal
paper-recipe SSL4RL models (Position, Rotation, Jigsaw, Contrastive) on:

- `MathVista_MINI`, the labeled 1,000-example MathVista mini test set supported
  by the pinned VLMEvalKit;
- `MMMU_DEV_VAL`, the labeled development/validation set. `MMMU_TEST` is not
  used because its labels are not locally reportable.

This is chain-of-thought evaluation, not answer-only inference. MathVista gets
an explicit `<think>...</think>` followed by `<answer>...</answer>` prompt.
MMMU uses the repository's patched MCQ reasoning prompt with the same tags.
Every raw response remains in the prediction workbook, and the run manifest
records non-empty and correctly formatted reasoning coverage.

MathVista's official pinned evaluator requires an auxiliary answer-extraction
judge. This project uses the official DeepSeek API with `deepseek-chat` as
required by the experiment protocol; MMMU is scored with `exact_matching` and
does not depend on that judge. The setup patch adds a distinct
`deepseek-chat` route because the pinned VLMEvalKit name `deepseek` refers to
SiliconFlow rather than the official DeepSeek endpoint. It also redacts API keys
from VLMEvalKit logs. For the trusted four-task handoff, the credential is
Base64-encoded in `deepseek_api_key_b64.py` and decoded only in process memory;
the plaintext is never written to configs, logs, manifests, or temporary files.
`python scripts/evaluation/smoke_reasoning_judges.py` validates a synthetic
MathVista extraction through the live DeepSeek API plus a synthetic MMMU exact
match without loading a vision-language model.

Download and verify both reasoning datasets:

```bash
python scripts/evaluation/prepare_reasoning_data.py \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval"
python scripts/evaluation/prepare_reasoning_data.py \
  --lmu-data-root "$FVE_OUTPUT_ROOT/inputs/vlmeval" --verify-only
```

The FVE Rotation model path is the `export.exported_model` value in the
successful fingerprinted `post_training_manifest.json`. Normal four-task
checkpoints must first be exported with `--artifact-type ssl4rl_export` and the
correct `--model-role`; task identities cannot be relabeled at evaluation time.

```bash
python scripts/evaluation/run_reasoning_benchmarks.py \
  --model-path <export.exported_model> --model-name Rotation --model-role rotation \
  --run-id <training-run-id> --gpu <one-gpu-index> \
  --output-root "$FVE_OUTPUT_ROOT" --mathvista-judge deepseek-chat
```

Each immutable attempt is written under
`$FVE_OUTPUT_ROOT/evaluation/reasoning/<run-id>/<model>-<fingerprint>/`.
For the four normal task models, repeat export and evaluation with the matching
role, then compare only compatible attempts:

```bash
python scripts/evaluation/aggregate_reasoning_results.py \
  --position-manifest <position-reasoning-manifest> \
  --rotation-manifest <rotation-reasoning-manifest> \
  --jigsaw-manifest <jigsaw-reasoning-manifest> \
  --contrastive-manifest <contrastive-reasoning-manifest> \
  --output-dir "$FVE_OUTPUT_ROOT/evaluation/reasoning_tables/four-task"
```

The resulting CSV/JSON reports each supplied model's MathVista/MMMU percentages
and reasoning-format coverage. Rotation-only aggregation remains backward
compatible. The default response budget is
4,096 tokens and uses deterministic sampling (`temperature=0.01`, `top-p=0.001`,
`top-k=1`).
