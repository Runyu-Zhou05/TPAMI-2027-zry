# Position-to-Golden RLVR

This workflow starts a new, unfrozen Golden-RLVR stage from the final normal
Position checkpoint. It is separate from the Rotation-only frozen-vision queue:
do not place it in that queue, relabel an FVE checkpoint as Position, or resume
an earlier Golden run.

## Fixed protocol

- Source data: the old `MMBench_DEV_EN` TSV used by the paper protocol, with MD5
  `b6caf1133a01c6bb705cf753bb527ed8`. Do not substitute
  `MMBench_DEV_EN_V11`.
- Split: seed 42, stratified by `l2-category`, with every 2-, 3-, or 4-example
  circular group kept intact. The generated files contain exactly 2,607
  training rows and 1,722 held-out test rows.
- Initialization: a standard Hugging Face export of the final **normal
  Position** run. Its `ssl4rl_export_manifest.json` must identify
  `artifact_type=ssl4rl_export` and `model_role=position`.
- Training: eight GPUs, 512 prompts per update, five rollouts per prompt, 20
  epochs, and 100 expected optimizer steps. Vision remains trainable.
- Validation and checkpoints: validation at step 0; validation plus a checkpoint
  at steps 20, 40, 60, 80, and 100. Automatic resume is disabled.

The 100-step expectation follows the trainer's drop-last behavior:
`floor(2607 / 512) = 5` updates per epoch.
Overlong-prompt filtering is disabled so the row budget cannot silently change;
`truncation=error` makes any prompt beyond 1,024 tokens fail closed. Preserve
the printed preflight budget in the run log and stop if the observed step
schedule differs.

## 1. Prepare the old MMBench split

Use the checksum-pinned old TSV already prepared for evaluation. Keep the
Golden dataset under the durable input root:

```bash
export FVE_OUTPUT_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
export FVE_DATA_ROOT="$FVE_OUTPUT_ROOT/inputs/datasets"
export MMBENCH_OLD_TSV="$FVE_OUTPUT_ROOT/inputs/vlmeval/MMBench_DEV_EN.tsv"

python3 scripts/golden/prepare_mmbench_golden_data.py \
  --source-tsv "$MMBENCH_OLD_TSV" \
  --output-dir "$FVE_DATA_ROOT/MMBench_GoldenQA"
python3 scripts/golden/prepare_mmbench_golden_data.py \
  --source-tsv "$MMBENCH_OLD_TSV" \
  --output-dir "$FVE_DATA_ROOT/MMBench_GoldenQA" \
  --verify-only
```

The required outputs are:

```text
$FVE_DATA_ROOT/MMBench_GoldenQA/train.parquet  # 2,607 rows
$FVE_DATA_ROOT/MMBench_GoldenQA/test.parquet   # 1,722 rows
$FVE_DATA_ROOT/MMBench_GoldenQA/manifest.json
```

The manifest binds the source checksum, exact train/test indices, complete
circular groups, generator identity, schemas, row counts, and output hashes.
`train.parquet` is the only optimization input; `test.parquet` is held out for
validation.

## 2. Supply the final Position HF export

Export the final committed checkpoint of the normal unfrozen Position run with
the existing evaluator exporter. Use the true Position role:

```bash
python3 scripts/evaluation/export_fve_checkpoint.py \
  --checkpoint <position-checkpoint-root> \
  --artifact-type ssl4rl_export \
  --model-role position \
  --result-json "$FVE_OUTPUT_ROOT/evaluation/exports/position-export-result.json"
```

Set `POSITION_HF_MODEL` to the exported model directory reported by that JSON.
The Golden launcher rehashes the export and rejects missing, modified,
non-Position, or FVE-labeled artifacts.

## 3. Inspect a pure dry-run

Choose a brand-new run ID and verify paths and overrides before allocating the
job. Dry-run does not read model/data artifacts and must not create any file or
directory.

```bash
export POSITION_HF_MODEL=<exported-position-hf-directory>
export GOLDEN_RUN_ID=<new-unique-run-id>

python3 scripts/golden/run_mmbench_golden.py \
  --model-path "$POSITION_HF_MODEL" \
  --data-root "$FVE_DATA_ROOT" \
  --output-root "$FVE_OUTPUT_ROOT" \
  --run-id "$GOLDEN_RUN_ID" \
  --gpus 0,1,2,3,4,5,6,7 \
  --dry-run
```

Confirm the printed command contains `trainer.total_epochs=20`,
`trainer.total_training_steps=100`,
`trainer.save_freq=20`, `trainer.test_freq=20`,
`trainer.val_before_train=True`, and `trainer.resume_mode=disable`. It must
announce step 0 validation and steps 20/40/60/80/100 for validation and
checkpoints.

## 4. Start one new production run

Remove only `--dry-run` from the inspected command:

```bash
python3 scripts/golden/run_mmbench_golden.py \
  --model-path "$POSITION_HF_MODEL" \
  --data-root "$FVE_DATA_ROOT" \
  --output-root "$FVE_OUTPUT_ROOT" \
  --run-id "$GOLDEN_RUN_ID" \
  --gpus 0,1,2,3,4,5,6,7
```

The launcher refuses a nonempty run directory and writes
`golden_preflight.json` before training. Outputs live under:

```text
$FVE_OUTPUT_ROOT/runs/$GOLDEN_RUN_ID/ssl4rl_qwen_3b_mmbench_position_to_golden/
```

Preserve the preflight record, console log, TensorBoard events, reward traces,
validation generations, and committed checkpoints. Expected milestones are:

| Step | Expected event |
|---:|---|
| 0 | held-out validation before training |
| 20 | validation and checkpoint |
| 40 | validation and checkpoint |
| 60 | validation and checkpoint |
| 80 | validation and checkpoint |
| 100 | final validation and checkpoint |

Summarize the validation dumps after successful completion:

```bash
python3 scripts/golden/summarize_golden_validation.py \
  --validation-data-dir \
    "$FVE_OUTPUT_ROOT/runs/$GOLDEN_RUN_ID/ssl4rl_qwen_3b_mmbench_position_to_golden/validation_generations" \
  --output-dir \
    "$FVE_OUTPUT_ROOT/evaluation/golden_validation/$GOLDEN_RUN_ID"
```

The launcher refuses success unless committed step 100 and validation dumps for
steps 0/20/40/60/80/100 all exist.

Do not enable resume or reuse a prior run ID. A failed or interrupted attempt
must be investigated in place and followed, when appropriate, by a new run ID;
the fixed reviewer protocol is a fresh Position-to-Golden run.
