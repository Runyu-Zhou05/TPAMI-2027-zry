# SEED-Bench Position-to-Golden RLVR

This workflow trains a new benchmark-specific Golden-RLVR model on the 60%
training partition of `SEEDBench_IMG`, initialized from the final normal
**SEED-Position** model. It is independent of the MMBench Golden run and the
Rotation-only frozen-vision queue. Never resume a Position run as Golden,
reuse a Golden run ID, or relabel an FVE checkpoint as Position.

## Fixed protocol

- Source: the pinned `SEEDBench_IMG.tsv` used by the six-benchmark evaluation,
  MD5 `68017231464752261a2526d6ca3a10c0`.
- Source schema: `answer`, `question`, `A`, `B`, `C`, `D`, `image`, `index`,
  and `category`, in that order; exactly 14,232 rows and nine
  image-understanding categories. Answers are single letters in `A`-`D`, and
  every image is strict-decodable JPEG Base64 without a `data:` prefix.
- Split: seed 42, independently rank rows within `category`, then assign 60%
  to training and 40% to held-out testing using deterministic SHA256 order and
  half-up rounding. A source row occurs in exactly one partition. The outputs
  contain exactly 8,539 training rows and 5,693 held-out test rows.
- Initialization: a standard Hugging Face export of the final **normal
  SEED-Position** run. Its `ssl4rl_export_manifest.json` must say
  `artifact_type=ssl4rl_export` and `model_role=position`.
- Training: eight GPUs, 512 prompts per update, five rollouts per prompt, 20
  epochs, and 320 optimizer steps. Vision remains trainable.
- Validation/checkpoints: validation at step 0; validation and a checkpoint
  every 20 steps through step 320. Automatic resume is disabled.

The step budget follows the trainer's drop-last behavior:
`floor(8539 / 512) = 16` updates per epoch and `16 * 20 = 320` total updates.
Overlong-prompt filtering is disabled so it cannot silently reduce this budget;
`truncation=error` fails closed if a prompt exceeds 1,024 tokens.

The exact stratified partition is:

| Category | Total | Train | Test |
|---|---:|---:|---:|
| Instance Attributes | 4,649 | 2,789 | 1,860 |
| Instance Identity | 1,831 | 1,099 | 732 |
| Instance Interaction | 97 | 58 | 39 |
| Instance Location | 978 | 587 | 391 |
| Instances Counting | 2,447 | 1,468 | 979 |
| Scene Understanding | 3,158 | 1,895 | 1,263 |
| Spatial Relation | 657 | 394 | 263 |
| Text Understanding | 84 | 50 | 34 |
| Visual Reasoning | 331 | 199 | 132 |
| **Total** | **14,232** | **8,539** | **5,693** |

## 1. Prepare the deterministic SEED-Bench split

Reuse the checksum-verified TSV prepared by the evaluation workflow:

```bash
export FVE_OUTPUT_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
export FVE_DATA_ROOT="$FVE_OUTPUT_ROOT/inputs/datasets"
export SEEDBENCH_TSV="$FVE_OUTPUT_ROOT/inputs/vlmeval/SEEDBench_IMG.tsv"

python3 scripts/golden/prepare_seedbench_golden_data.py \
  --source-tsv "$SEEDBENCH_TSV" \
  --output-dir "$FVE_DATA_ROOT/SEEDBench_GoldenQA"
python3 scripts/golden/prepare_seedbench_golden_data.py \
  --source-tsv "$SEEDBENCH_TSV" \
  --output-dir "$FVE_DATA_ROOT/SEEDBench_GoldenQA" \
  --verify-only
```

The required outputs are:

```text
$FVE_DATA_ROOT/SEEDBench_GoldenQA/train.parquet  # 8,539 rows
$FVE_DATA_ROOT/SEEDBench_GoldenQA/test.parquet   # 5,693 rows
$FVE_DATA_ROOT/SEEDBench_GoldenQA/manifest.json
```

The manifest binds the pinned TSV checksum, exact train/test indices,
category-stratified split, generator identity, schemas, row counts, and output
hashes. Only `train.parquet` is optimized; `test.parquet` remains held out for
validation. Do not reuse the separate detail-evaluation TSV as training data.

## 2. Export the final normal SEED-Position checkpoint

Export a committed checkpoint from the normal, unfrozen SEED-Position run. The
launcher requires the export manifest's actor checkpoint path to identify both
`seed` and `position` and rejects frozen-vision exports:

```bash
python3 scripts/evaluation/export_fve_checkpoint.py \
  --checkpoint <position-checkpoint-root> \
  --artifact-type ssl4rl_export \
  --model-role position \
  --result-json "$FVE_OUTPUT_ROOT/evaluation/exports/position-export-result.json"
```

Set `POSITION_HF_MODEL` to the `exported_model` directory in that result. The
SEED Golden launcher validates and hashes the export. This path initializes
both actor and reference policy; `trainer.resume_mode=disable` creates a fresh
optimizer, scheduler, dataloader, and global step.

## 3. Inspect a pure dry-run

Dry-run validates arguments and prints the complete command without reading
model/data artifacts or creating files and directories:

```bash
export POSITION_HF_MODEL=<exported-position-hf-directory>
export SEED_GOLDEN_RUN_ID=<new-unique-run-id>

python3 scripts/golden/run_seedbench_golden.py \
  --model-path "$POSITION_HF_MODEL" \
  --data-root "$FVE_DATA_ROOT" \
  --output-root "$FVE_OUTPUT_ROOT" \
  --run-id "$SEED_GOLDEN_RUN_ID" \
  --gpus 0,1,2,3,4,5,6,7 \
  --dry-run
```

Confirm the printed command contains `data.train_batch_size=512`,
`trainer.total_epochs=20`, `trainer.total_training_steps=320`,
`trainer.save_freq=20`, `trainer.test_freq=20`,
`trainer.val_before_train=True`, and `trainer.resume_mode=disable`. It must use
only `SEEDBench_GoldenQA/train.parquet` for training and the matching
`test.parquet` for validation.

## 4. Start one fresh production run

Remove only `--dry-run` from the inspected command:

```bash
python3 scripts/golden/run_seedbench_golden.py \
  --model-path "$POSITION_HF_MODEL" \
  --data-root "$FVE_DATA_ROOT" \
  --output-root "$FVE_OUTPUT_ROOT" \
  --run-id "$SEED_GOLDEN_RUN_ID" \
  --gpus 0,1,2,3,4,5,6,7
```

The launcher refuses a nonempty run directory and writes its preflight record
before training. Preserve that record, the console log, TensorBoard events,
reward traces, validation generations, and every committed checkpoint.

Expected validation steps are:

```text
0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320
```

Steps 20 through 320 also produce checkpoints. The launcher reports success
only when committed step 320 and every validation dump above exist.

After success, reuse the shared validation summarizer with the SEED schedule:

```bash
python3 scripts/golden/summarize_golden_validation.py \
  --validation-data-dir \
    "$FVE_OUTPUT_ROOT/runs/$SEED_GOLDEN_RUN_ID/ssl4rl_qwen_3b_seedbench_position_to_golden/validation_generations" \
  --output-dir \
    "$FVE_OUTPUT_ROOT/evaluation/seed_golden_validation/$SEED_GOLDEN_RUN_ID" \
  --expected-steps 0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320
```

Do not enable resume after an interruption. Investigate the failed attempt in
place and, when rerunning is justified, use a new run ID so the reviewer-facing
protocol remains a fresh Position-to-SEED-Golden stage.
