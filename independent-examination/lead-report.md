# Independent end-to-end code examination

Date: 2026-11-15
Repository: `TPAMI-2027-zry`
Scope: environment -> model -> data -> Rotation training -> reward trace -> checkpoint/HF export -> evaluation -> Tables I/II/III -> final package

## Verdict

**Do not start the production queue and do not report Tables I--III from the current pipeline.** The 8-GPU batch configuration and frozen-vision mechanics are wired coherently, but the current code has blocking resume/state-lineage failures and unresolved evaluation-protocol mismatches. In particular, a routine interruption can make checkpoint resume fail, resuming the terminal checkpoint performs an extra update, stale evaluation artifacts can be relabeled as a new model, RealWorldQA is routed through a generic MCQ class, and the packager accepts unrelated or invalid inputs without preserving the required acceptance chain.

This was a context-isolated code audit. No SSH, remote host, network access, GPU, actual model inference, dataset download, or smoke/training run was used. Findings were reproduced only with static inspection, CLI/dry-run checks, and temporary CPU synthetic fixtures.

## Confirmed release blockers

### 1. Training is not safely resumable after an ordinary interruption (High)

- **Evidence:** a reward trace is published before actor update and checkpoint publication (`verl/trainer/ppo/ray_trainer.py:1303-1317,1354-1361,1395-1410`), while existing step traces are never overwritten (`verl/utils/reward_trace.py:84-89`). Production checkpoints are spaced 200 steps apart (`scripts/frozen_vision/run_frozen_vision.py:74-77,167`).
- **Impact:** rollback to checkpoint K reruns K+1 but collides with the trace left by the interrupted attempt. Before checkpoint 200, any interruption after the first trace generally makes auto-resume fail at step 1.
- **Static reproduction:** tracker 200 plus an existing trace 201 necessarily reaches `FileExistsError` before the resumed update commits.
- **Required fix:** make traces attempt-aware or reconcile/quarantine all traces beyond the committed checkpoint before resume; add interruption tests at pre-checkpoint, mid-interval, and checkpoint-publication boundaries.

### 2. Resuming a completed checkpoint mutates the final model (High)

- **Evidence:** load restores `global_steps`, then `fit()` increments it unconditionally and enters the loop (`verl/trainer/ppo/ray_trainer.py:997-1052,1111-1135`). The actor update and new checkpoint occur before the `is_last_step` return (`:1181,1354-1361,1395-1445`).
- **Impact:** resuming terminal step 420 executes and saves step 421. A recovery attempt after a post-training/queue failure therefore changes the result and exceeds the configured budget.
- **Required fix:** return successfully when restored step equals the configured total, and reject a greater step unless explicit extension is requested.

### 3. Base-model and generated-data identity are not enforced before expensive work (High)

- **Evidence:** onboarding validates only Qwen model type; the runtime freeze audit prints but does not assert the required visual/trainable/total counts (`verl/workers/fsdp_workers.py:330-345`). `prepare_rotation_data.py:105-158` checks required column names, row counts, first-row source, and self-recorded hashes, but canonical generated hashes and full Arrow/nested schema/build invariants are not required.
- **Impact:** a wrong Qwen2.5-VL size/revision or a self-attested non-canonical dataset can pass preflight, fail late, or silently produce a non-reproducible run.
- **Synthetic reproduction:** a manifest with the pinned source/revision but altered seed, granularity, completion flag, and split map was accepted when its self-recorded file hashes matched.
- **Required fix:** add a fail-closed CPU model/data preflight called by the queue: exact architecture/counts/weight identity, exact Arrow schema and per-row task invariants, pinned raw hashes, generation parameters, split map, and canonical/trusted output provenance.

### 4. Evaluation can silently publish stale scores under a new checkpoint (Critical)

- **Evidence:** output identity is only `run_id/model_name`, the runner always passes `--reuse`, and `require_scores()` accepts one matching filename without model/config/data binding (`scripts/evaluation/run_six_benchmarks.py:32-43,88-119`). A new manifest then records the current model path (`:132-147`).
- **Impact:** reusing a label after changing/re-exporting a model can relabel old predictions and scores as the new model and feed them into all paper tables.
- **CPU reproduction:** a fake evaluator that exited zero plus six pre-existing `OLD_*` score artifacts caused the real wrapper to emit a new manifest claiming a different model while referencing all old scores.
- **Required fix:** immutable attempt directories keyed by model-weight/export, dataset, evaluator, patch, and decoding-config fingerprints; refuse a nonempty directory unless every fingerprint and artifact hash matches.

### 5. The six-benchmark protocol is not source-consistent or reportable (Critical)

- **RealWorldQA:** it is configured as `ImageMCQDataset` with the global exact-matching invocation (`scripts/evaluation/run_six_benchmarks.py:15-21,109-119`), despite being open-answer VQA. This must use the pinned kit's dataset-specific VQA class/evaluator and be tested against the real TSV schema.
- **Prompt/pixels:** repository `README.md:200-218` specifies a customized MCQ prompt and `min_pixels=35840`, while the setup applies only the Qwen path-detection patch and the runner uses `use_custom_prompt=False` and `min_pixels=1,003,520` (`scripts/evaluation/setup_vlmevalkit.sh:27-29`; `run_six_benchmarks.py:90-103`). The current handoff documents a different path but does not resolve this paper-recipe conflict.
- **V* identity:** current handoff code says standard VStarBench, while `README.md:131-140` lists MMStar among paper benchmarks and not VStar; prior tracked evaluation notes described an agentic V* protocol. Code alone does not establish which fills the target table column.
- **Impact:** even a successful run may evaluate the wrong task, prompt, preprocessing budget, or V* protocol.
- **Required fix:** bind every Table I column to an authoritative paper dataset release/split/evaluator/prompt/config, then encode and hash that protocol in the runner and manifest before inference.

### 6. Dataset and evaluator provenance is presence-only at evaluation time (High)

- **Evidence:** preparation records MD5/SHA-256, but the runner only checks keys and file existence (`scripts/evaluation/run_six_benchmarks.py:75-86`). Evaluator validation is only `.pinned-commit` text (`:67-70`); the run manifest contains mutable paths and no data/model/config/result hashes (`:132-147`).
- **Impact:** modified datasets, decoy manifest paths, or edited evaluator code can be reported as the pinned protocol.
- **CPU reproduction:** fake source/marker, corrupted TSVs, and decoy manifest paths passed runner preflight.
- **Required fix:** recompute pinned dataset digests at launch, enforce canonical `$LMUData` paths/schema/counts, verify Git HEAD plus allowlisted patch digest, and snapshot normalized provenance and artifact hashes into each run.

### 7. The 60/40 held-out Table II/III protocol is not implemented (High)

- **Evidence:** no script generates or manifests the stated split. Four score overrides are independently optional and raw CSVs are accepted without split/model/run compatibility (`scripts/evaluation/aggregate_paper_tables.py:69-100,135-170`).
- **Impact:** polished tables can mix full and held-out results, use different examples per model, or combine unrelated runs.
- **CPU reproduction:** mismatched run IDs/model labels/commit values were accepted; supplying only one MMBench override changed one side of Table II while the other three overrides remained absent.
- **Required fix:** deterministic split generator with exact indices and checksum; accept detail run manifests, not naked CSVs; require either zero overrides or a complete compatible four-manifest set.

### 8. Checkpoint/export commitment is not reliable (High)

- **Evidence:** the trainer directly truncates/writes `latest_checkpointed_iteration.txt` despite its atomic-use comment (`verl/trainer/ppo/ray_trainer.py:990-995`), and its reader directly parses it (`verl/utils/checkpoint/checkpoint_manager.py:167-196`). The exporter ignores that tracker and selects the largest actor directory with a config (`scripts/evaluation/export_fve_checkpoint.py:29-42`). Existing merged targets are not source-fingerprint-bound.
- **Impact:** tracker corruption blocks resume; a higher partial checkpoint masks the committed one; stale merged weights can be relabeled.
- **CPU reproduction:** committed step 200 plus partial step 201 caused the resolver to select step 201.
- **Required fix:** atomic, fsynced committed-checkpoint manifest after complete shards/data/HF artifacts; exporter must resolve and validate that exact committed step and bind cached exports to source hashes.

### 9. Final packaging does not validate or preserve the acceptance chain (Critical)

- **Evidence:** the packager checks only that four paths exist (`scripts/evaluation/package_final_results.py:88-101`) and never parses/links run IDs, model roles, table inputs, queue completion, post-training smoke, or data provenance. It omits the queue state, post-training manifest/smoke, and evaluation-data manifest from the curated bundle.
- **Impact:** a checksummed “final” archive can contain invalid/unrelated evidence and cannot prove the required workflow passed.
- **CPU reproduction:** the real packager exited zero with empty training/tables directories and two empty non-JSON manifest files; the tar contained those files and no acceptance evidence.
- **Required fix:** schema-validate and lineage-link every manifest; require queue success and post-training smoke; include the normalized data/evaluator/export/table provenance and hashes using relative references.

## Protocol decision requiring owner confirmation

Production deliberately passes both logical `train.parquet` and `test.parquet` into `data.train_files`, and uses the latter again for validation (`scripts/frozen_vision/run_frozen_vision.py:67-69,124-126`). The original `run_mmbench_rotation.sh:4-13` does the same, so this may be an intentional transductive paper protocol rather than an accidental regression. Nevertheless, the metric is not held out and the overlap with MMBench DEV evaluation must be resolved explicitly. Do not silently change it; either reserve test for validation or document/fingerprint the transductive protocol and stop calling that metric held out.

## Lower-severity correctness/reproducibility issues

- Mandatory environment gate 1 prints most versions but asserts only `sm_90`; it does not enforce the stated compatibility family.
- Data-root defaults disagree: preparation defaults below the output root, while downstream commands fall back to checkout-local `our_datasets`.
- Reward summaries do not validate schema, step contiguity, run/task identity, per-step response count, filename equality, or final-checkpoint agreement.
- Score-unit inference treats exactly `1.0` as fraction 100% and `1.01` as percent 1.01%, and does not reject NaN/Infinity/out-of-range values (`aggregate_paper_tables.py:39-66`).
- `run_six_benchmarks.py --dry-run` creates its work directory and config before returning (`:88-122`), contrary to repository-wide dry-run purity.
- Files over 512 MiB receive `sha256: null` in the final inventory, so omitted model/checkpoint shards have no cryptographic identity.

## Small fixes applied directly

1. **Exact smoke-data shape:** `scripts/frozen_vision/prepare_smoke_data.py:13-25` now rejects a source with fewer than the requested 10/512 rows and asserts the emitted row count before writing. This prevents an undersized “production-shape” gate from silently succeeding.
2. **Retryable package publication:** `scripts/evaluation/package_final_results.py:97-132` now builds and hashes the archive and sidecar in a temporary same-filesystem directory and publishes complete files with `os.replace`. An injected tar failure now leaves no final archive or sidecar, so retry is not poisoned.

No major semantic/protocol change was applied. In particular, evaluation class/protocol choices, transductive split usage, trace reconciliation, and manifest lineage need owner review and dedicated tests.

## Checks run and results

- Fully read the 397-line root `AGENTS.md` before action.
- Delegated four fresh-context audit tracks: onboarding; training/reward/checkpoints; evaluation/tables; packaging/recovery.
- `python3 -m compileall`/`py_compile` passed for all focused workflow and modified files.
- AST parsed all 15 frozen-vision/evaluation Python scripts.
- CLI `--help` passed for all stdlib-only workflow entry points.
- 8-GPU production dry run: rollout batch 2,560; PPO mini-batch 640; per-GPU 80; micro 10; no output root created.
- 8-GPU quick dry run: rollout batch 16; PPO mini-batch 16; per-GPU 2; micro 1.
- Synthetic smoke test: a 3-row source requested as 512 rows is rejected and no output is written.
- Package success fixture: tar readable and SHA-256 sidecar verifies.
- Injected package failure: no final archive or sidecar remains.
- Incomplete-package fixture: reproduced pre-existing design gap (packager still accepts empty/invalid workflow inputs; reported above).
- Child CPU fixtures reproduced stale evaluation reuse, corrupted-data/fake-marker acceptance, partial held-out override acceptance, and partial-checkpoint selection.
- Repository pytest could not run in the existing offline local interpreter because pytest, Torch, PyArrow, Transformers, TorchData, and OmegaConf are absent. No dependencies were installed under the examination constraints.

## Report files

- `independent-examination/child-01-onboarding.md`
- `independent-examination/child-02-training-checkpoints.md`
- `independent-examination/child-03-evaluation-tables.md`
- `independent-examination/child-04-packaging-recovery.md`
- `independent-examination/lead-report.md`
