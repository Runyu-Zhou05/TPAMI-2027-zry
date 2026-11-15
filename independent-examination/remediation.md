# Remediation of the independent workflow examination

This file records the disposition of the independent examination completed on
2026-11-15. It is a code-review record, not evidence that the unavailable H20
production job has run.

## Resolved blockers

- Rotation optimization now reads only `train.parquet`; `test.parquet` is
  validation-only. The generated-data manifest, exact schema, semantics, raw
  source hashes, generator hash, and pinned model revision are revalidated
  before production startup.
- Checkpoints publish a completion manifest before an atomic tracker update.
  Schema-v2 manifests hash every actor/critic/data file; resume and export
  recompute those hashes. Resume quarantines reward traces beyond the committed
  step, rejects over-budget checkpoints, and returns without an extra update at
  the terminal step.
- Reward summaries require the full per-rollout schema, exact run/task identity,
  expected rows per step, contiguous steps, valid timestamps, and agreement
  with the committed checkpoint.
- HF export follows the committed tracker rather than the highest directory,
  validates all recorded files, binds cached exports to the source completion
  manifest, and records every safetensor digest.
- Full and detail evaluation attempts are immutable fingerprints of model
  weights and loading metadata, exact data bytes, the evaluator Python tree,
  environment package freeze, and decoding config. Completed attempts revalidate
  preflight/config, prediction/score hashes, and exact prediction-index coverage
  before reuse.
- Tables II/III now use a deterministic manifest-backed 60/40 split. MMBench
  circular groups stay intact, SEED is category-stratified, and the aggregator
  accepts either no detail manifests or one compatible Base/Rotation pair.
  Split reuse and final packaging regenerate the exact partition from the
  checksum-pinned full datasets and reject self-attested or modified subsets.
- Queue success is atomic JSON containing invocation and per-stage evidence.
  Post-training smoke is attempt-fingerprinted. Final packaging validates the
  queue, every stage checkpoint/reward summary, Rotation data, final export,
  paired full/detail evaluation, and table lineage; all omitted large files are
  stream-hashed. Final paper packaging requires the paired deterministic 40%
  detail runs.

## Protocol decisions resolved by the user/source

- The README value `min_pixels=35840` is treated as a missing-`* 28`
  documentation typo. README and runners now use `1280 * 28 * 28 = 1,003,520`.
- The upstream SSL4RL MCQ answer-format prompt is applied idempotently to the
  pinned VLMEvalKit source.
- The paper PDF states that each downstream benchmark is divided into 60%
  training and 40% held-out test data for the golden-reward comparison. It does
  not publish a seed. This repository therefore declares seed 42 and records
  the exact split instead of claiming recovery of an unspecified original seed.
- RealWorldQA is intentionally handled by `ImageMCQDataset` in the pinned
  VLMEvalKit because that checksum-pinned TSV contains answer choices.
- V* is the paper's standard VStarBench MCQ evaluation, not a separate
  crop/tool-agent loop.

## Verification boundary

Ten CPU-only contract tests cover circular-group isolation, deterministic SEED
splitting, rejection of self-attested splits, committed-checkpoint export
selection, actual export-weight rehashing, prompt-patch idempotence, paired
table inputs and explicit score units, and complete synthetic final packaging.
Python compilation, diff whitespace checks, and the 8-GPU launch dry-run pass
locally. Real CUDA quick/quick-resume/production-shape gates and the six
benchmark runs remain mandatory on the colleague's machine.
