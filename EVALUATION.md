# FVE post-training evaluation

## Evaluation objective

After Rotation FVE training, evaluate the exported model on the original VQA
tasks rather than on Rotation-generated questions:

- MMBench;
- SEED-Bench;
- V*Bench with the agentic visual-search protocol;
- RealWorldQA;
- BLINK;
- MME.

MMBench must report Logical, Relation, Attribute, Coarse, Cross-Instance,
Single-Instance, and Average. SEED-Bench must report the nine subtasks shown in
the paper table plus Average. Table I uses the six benchmark-level scores and
their unweighted average.

The existing `build_benchmarks/makerotation_*.py` files construct SSL4RL
Rotation tasks from benchmark images. They are not the post-training benchmark
evaluation implementation. In particular, `makerotation_mme.py` uses
MME-RealWorld-Lite, not the standard MME benchmark required here.

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
  -> separate V*Bench agentic visual-search runner
  -> per-sample predictions and official detail files
  -> Table I, Table II, and Table III aggregation
```

Before any full benchmark run, repeat the same MMBench smoke with an exported
quick FVE checkpoint. This is the acceptance test for the checkpoint exporter.
Use `scripts/evaluation/run_post_training_smoke.py` to perform checkpoint
resolution, export validation, and MMBench smoke as one guarded command.
