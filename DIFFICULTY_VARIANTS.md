# Table XI harder-variant training

This workflow reproduces the four hard counterparts of the completed MMBench
visual SSL tasks:

| Task | Easy | Hard dataset |
|---|---|---|
| Rotation | 90 degrees | `MMBench_RotationQA_45degree` |
| Jigsaw | 2x2 | `MMBench_JigsawQA_3x3` |
| Contrastive | weak augmentation | `MMBench-ContrastiveQA_strong` |
| Position | 2x2 | `MMBench-PositionQA_3x3` |

The normal unfrozen Qwen2.5-VL-3B paper recipe is retained. The original
launchers optimize on both generated MMBench splits and also use the generated
test split for validation. This transductive behavior is kept so hard variants
remain directly comparable to the completed easy variants.

## 1. Generate raw variants

Run with the same raw MMBench installation used for the easy generators:

```bash
python3 build_benchmarks/makerotation_mmbench.py --granularity 45
python3 build_benchmarks/makejigsaw_mmbench.py --grid-size 3
python3 build_benchmarks/makecontrastive_mmbench.py --augmentation strong
python3 build_benchmarks/makeposition_mmbench.py --grid-size 3
```

Defaults remain backward-compatible with the easy variants. Hard outputs use
separate directories and never overwrite easy datasets.

## 2. Convert to the verl RL schema

Repeat this pair for each of `rotation45`, `jigsaw3x3`,
`contrastive_strong`, and `position3x3`:

```bash
python3 scripts/difficulty/prepare_hard_variant_data.py \
  --task <task> --raw-root <generated-data-parent> \
  --output-root <processed-data-root>
python3 scripts/difficulty/prepare_hard_variant_data.py \
  --task <task> --raw-root <generated-data-parent> \
  --output-root <processed-data-root> --verify-only
```

Each processed directory contains `train.parquet`, `test.parquet`, and a
manifest. Preprocessing verifies expected image counts (nine for Jigsaw3x3,
two otherwise), prompt placeholders, reward identity, and split.

## 3. Dry-run and one-step smoke

```bash
python3 scripts/difficulty/run_mmbench_hard.py \
  --task <task> --mode smoke --run-id <unique-smoke-id> \
  --model-path <pinned-base-model> --data-root <processed-data-root> \
  --output-root "$FVE_OUTPUT_ROOT" --gpus 0,1,2,3,4,5,6,7 --dry-run
```

Remove only `--dry-run` for one real optimizer step. Jigsaw3x3 explicitly sets
the vLLM image limit to nine. Do not launch full training before all four
smokes pass.

## 4. Full training

```bash
python3 scripts/difficulty/run_mmbench_hard.py \
  --task <task> --mode train --run-id <unique-train-id> \
  --model-path <pinned-base-model> --data-root <processed-data-root> \
  --output-root "$FVE_OUTPUT_ROOT" --gpus 0,1,2,3,4,5,6,7
```

Production uses 512 prompts, five rollouts, 20 epochs, checkpoint frequency
50, validation frequency 200, and no frozen visual modules. Per-rollout reward
tracing is enabled by default. Each task has an independent run directory.

After training, run the existing reward-difficulty analyzer on every task trace
with `--expected-rollouts-per-prompt 5`. Compare easy and hard variants using
informative-group fraction, reward entropy, success rate, and Table-XI
downstream changes.
