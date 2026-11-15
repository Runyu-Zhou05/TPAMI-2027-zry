# TPAMI FVE remediation closure review

Date: 2026-11-15

## Verdict

**PASS for code-only closure.** The previously confirmed blockers are closed in the current tree. No remaining high-impact static or CPU-reproducible path was found that silently trains the wrong split/model, evaluates stale or mismatched artifacts, or packages an inconsistent lineage.

This verdict does **not** claim target-machine readiness or reportable benchmark results. The mandatory CUDA, FlashAttention, quick, quick-resume, production-shape, post-training smoke, and six-benchmark runs remain required.

## Closed blocker map

| Area | Closure evidence |
|---|---|
| train-only | Production `data.train_files` contains only `train.parquet`; test remains validation-only. |
| checkpoint commit / terminal resume | Atomic tracker follows a SHA-256 completion manifest; corrupt state fails closed; terminal step performs no extra update. |
| reward rollback | Uncommitted traces are quarantined and summaries must agree with the committed checkpoint. |
| model/data identity | Pinned snapshot ETags/files and canonical Rotation output hashes are enforced before training. |
| committed export | Export follows the tracker, validates checkpoint hashes, exact parameter counts, and current exported model/support bytes. |
| immutable evaluation | Full/detail fingerprints bind actual bytes, evaluator tree, environment, data, and config; resume revalidates preflight/config/artifacts. |
| 60/40 split | MMBench group-aware and SEED-stratified partitions are regenerated and verified from checksum-pinned sources. |
| table pairing | Authenticated Base/Rotation full and paired detail manifests, prediction/score binding, units, and artifact hashes are enforced. |
| queue/post/final lineage | Exact stage snapshots, post-smoke artifacts, model bytes, detail split, table inputs, and final archive are verified; publication is locked and atomic per file. |

## Validation performed

- `python3 -m unittest -v tests.workflow.test_fve_workflow_contracts`: **10/10 passed**.
- Focused `py_compile`, `bash -n`, and `git diff --check`: passed.
- 8-GPU train dry-run: rollout 2560, PPO mini-batch 640, per-GPU 80, micro 10; train/test separation confirmed; no filesystem output.
- 8-GPU quick dry-run: rollout/PPO batch 16, per-GPU 2, micro 1; no filesystem output.
- Additional CPU negative fixtures covered base snapshot mutation, same-size checkpoint mutation, exported-weight mutation, fake detail split, stage-summary overwrite, and tampered post-smoke reuse.

The local base interpreter lacks the full Torch/PyArrow/Transformers/Ray training environment, so GPU-dependent repository tests and real inference were intentionally not attempted or replaced with downloads.

## Reports

- `closure-child-training.md`
- `closure-child-identity-export.md`
- `closure-child-evaluation.md`
- `closure-child-packaging.md`
- `closure-review.md`
