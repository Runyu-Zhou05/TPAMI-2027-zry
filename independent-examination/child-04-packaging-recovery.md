# Independent examination: packaging, durable state, and failure recovery

Scope was limited to static inspection and CPU-only temporary fixtures. No SSH,
remote host, network, GPU, model inference, dataset download, or smoke run was
used.

## Confirmed issues

### 1. High: a rerun can publish stale benchmark scores as results for a new model

- **Evidence:** `scripts/evaluation/run_six_benchmarks.py:32-43` accepts the one
  matching score file already present for each benchmark without checking when
  or by which model it was produced. The output directory is deterministically
  reused from only `run_id/model_name` (`:88-89`), and every invocation passes
  VLMEvalKit `--reuse` (`:109-119`). After the subprocess exits zero, the script
  writes a new manifest containing the current `model_path` but paths to any
  pre-existing matching scores (`:128-147`); there is no prior-manifest or
  model/config/data fingerprint comparison.
- **Impact:** reusing a run ID and label after changing or re-exporting the
  Rotation model can silently relabel old predictions and scores as the new
  model's results. Those scores then feed all three paper tables and the final
  archive.
- **Reproduction/static test:** a temporary fake pinned VLMEvalKit whose
  `run.py` only exited zero was paired with six files named
  `OLD_<benchmark>_acc.csv`/`_rating.json` under an existing
  `same-run/Rotation` directory. The real wrapper exited 0 and created a new
  manifest claiming `new-model` while listing all six `OLD_*` artifacts.
- **Recommended fix:** give every evaluation attempt a staging directory keyed
  by a content fingerprint of model weights/export manifest, evaluation-data
  hashes, VLMEval commit, and decoding config. Reuse only when a prior manifest
  has the identical fingerprint and verifies every artifact hash; otherwise
  refuse or quarantine the old directory. Atomically publish the completed
  attempt and record prediction and score hashes.

### 2. High: “latest checkpoint” selection can choose an uncommitted partial checkpoint

- **Evidence:** the trainer writes actor state, dataloader state, and only then
  `latest_checkpointed_iteration.txt` (`verl/trainer/ppo/ray_trainer.py:939-995`);
  its resume reader treats that tracker as the committed checkpoint
  (`verl/utils/checkpoint/checkpoint_manager.py:167-196`). In contrast,
  `scripts/evaluation/export_fve_checkpoint.py:29-42` ignores the tracker and
  selects the numerically largest `global_step_*` having only an `actor/`
  directory and HF `config.json`. A terminated save can satisfy that weak test
  before the checkpoint is committed. In addition, an already existing merged
  export directory skips merging altogether (`:142-167`) without checking that
  it was produced from the selected actor checkpoint.
- **Impact:** post-training export can fail on a partial newest directory even
  though a valid committed checkpoint exists, or can validate/relabel stale
  merged weights. This blocks recovery and can disconnect evaluation from the
  checkpoint recorded in the result manifest.
- **Reproduction/static test:** a temporary checkpoint root contained committed
  `global_step_1`, a tracker containing `1`, and a partial `global_step_2` with
  only `actor/huggingface/config.json`. Calling the real `resolve_actor_dir`
  printed `tracker=global_step_1` but selected `global_step_2`.
- **Recommended fix:** resolve directory roots exclusively through a strictly
  parsed, atomically written tracker, then verify all expected actor shards/HF
  files and dataloader state before export. Require an explicit step for any
  fallback. For an existing merged target, verify a source-checkpoint
  fingerprint and weight hashes before reuse; otherwise rebuild in staging and
  atomically replace it.

### 3. High: an interrupted package leaves a poisoned final archive that blocks retry

- **Evidence:** `scripts/evaluation/package_final_results.py:97-100` refuses any
  existing final archive. It then opens that final pathname directly for
  writing (`:113-122`), so creation happens before all members are added. The
  archive is hashed and the sidecar is written only afterwards (`:124-126`),
  also directly rather than transactionally.
- **Impact:** process death, storage exhaustion, an unreadable source, or a
  checksum/sidecar write failure can leave an incomplete archive (or an archive
  without a sidecar). The documented command is neither retry-safe nor
  idempotent: every later run stops at the existence check, requiring manual
  deletion with no built-in validation of what is safe to discard.
- **Reproduction/static test:** a CPU fixture injected an exception from
  `TarFile.add` after the final pathname was created. A 15-byte partial archive
  remained, and an unmodified second `main()` invocation failed with
  `Refusing to overwrite result archive`.
- **Recommended fix:** write archive and sidecar to unique sibling temporary
  paths, close and reopen/test the tar, compute and verify its SHA-256, fsync as
  appropriate, then atomically rename both into place. If final outputs exist,
  return success only when the sidecar and expected input fingerprint match;
  otherwise fail with a supported `--replace-invalid`/quarantine recovery path.

### 4. High: the final package neither enforces nor preserves the required acceptance chain

- **Evidence:** the packager only checks that four argument paths exist
  (`scripts/evaluation/package_final_results.py:88-100`); it never parses either
  evaluation manifest, checks their `run_id`/model identities, connects the
  tables to them, or requires queue/post-training success. It copies only the
  two full-evaluation parent directories, the tables directory, and selected
  training files (`:60-72,102-120`). Under the documented layout this omits
  `state/<run-id>/queue.complete`,
  `evaluation/post_training/<run-id>/post_training_manifest.json` and its smoke
  artifacts, and `inputs/vlmeval/six_benchmark_manifest.json`, even though
  `AGENTS.md:291-322` defines the queue audit and post-training manifest as gates
  for the long evaluation. The included full-evaluation manifest contains only
  an absolute path to the omitted data manifest
  (`run_six_benchmarks.py:132-142`).
- **Impact:** a checksummed archive can look final while containing unrelated
  runs, invalid manifests, no evidence that the required post-training smoke
  passed, and no portable record of the evaluated dataset hashes. Off the
  source machine, its absolute provenance links are broken.
- **Reproduction/static test:** the real packager exited 0 when given a claimed
  run ID, a training directory and tables from differently named runs, and two
  evaluation “manifests” containing the literal text `not even json`. Tar
  inspection confirmed that the invalid manifests were included while the
  synthetic post-training manifest, queue marker, and evaluation-data manifest
  were all absent.
- **Recommended fix:** accept an output root plus run ID (or a single signed
  root manifest), load and schema-validate every dependency, require exact run
  and model lineage, verify the queue marker and post-training smoke manifest,
  and check that table inputs equal the supplied evaluation manifests. Copy the
  queue state, post-training export/smoke evidence, evaluation-data manifest,
  decoding config, and their hashes into the bundle using relative references.

### 5. High: full evaluation trusts dataset existence but ignores its recorded checksums

- **Evidence:** `prepare_evaluation_data.py:57-92,106-142` computes MD5 and
  SHA-256 values and stores them in `six_benchmark_manifest.json`.
  `run_six_benchmarks.py:75-86` reads that manifest but checks only that all six
  keys and referenced files exist; it does not compare either digest, byte
  count, URL, or the pinned checksum table. Its run manifest later records only
  the mutable manifest's path (`:132-142`).
- **Impact:** any dataset modified or replaced after preparation is silently
  accepted, so a successful run and final paper tables need not correspond to
  the pinned six-benchmark protocol.
- **Reproduction/static test:** the stale-score fixture supplied six arbitrary
  text files whose manifest advertised `declared-but-not-checked` for both
  digests. The wrapper passed dataset preflight and exited 0.
- **Recommended fix:** before launching VLMEvalKit, recompute each dataset MD5
  and SHA-256 and compare with both the manifest and the repository-pinned
  specification. Embed a normalized copy and digest of the data manifest in
  each run manifest and final bundle, rather than only its absolute path.

### 6. Medium: durable success is presence-only and can survive as stale or corrupt state

- **Evidence:** both the queue and watcher treat any existing
  `queue.complete` as success without reading it
  (`scripts/frozen_vision/run_queue.py:58-69`;
  `scripts/frozen_vision/gpu_watcher.py:71-82`). Queue failure and success
  markers are written directly (`run_queue.py:106-112`), with no invocation
  fingerprint or per-stage records. Likewise, post-training and full-evaluation
  output directories are reused without removing/invalidating an older success
  manifest before a new attempt
  (`scripts/evaluation/run_post_training_smoke.py:49-53,96-111`;
  `scripts/evaluation/run_six_benchmarks.py:88-89,128-147`).
- **Impact:** a zero-byte/partially written queue marker makes a later command
  report success before checking model, data, disk, logs, summaries, or
  checkpoints. A failed rerun can leave an older post-training/evaluation
  success manifest visible beside a newly truncated log, making filesystem-only
  automation misclassify the attempt.
- **Reproduction/static test:** a temporary output root with a zero-byte
  `state/marker-case/queue.complete` and a nonexistent model was passed to the
  real queue command. It exited 0 and printed `Queue already completed`; the
  marker remained zero bytes.
- **Recommended fix:** represent attempts as atomic JSON state containing
  schema version, argument/config fingerprint, start/end times, stage status,
  checkpoint step, and required artifact hashes. Publish success only after all
  checks pass via temp-file + fsync + `os.replace`; validate content on reuse and
  fail closed on corruption. Stage new reruns in attempt-specific directories
  and only update a `current-success` pointer atomically after completion.

### 7. Medium: large omitted artifacts have no cryptographic identity in the inventory

- **Evidence:** `scripts/evaluation/package_final_results.py:43-55` records
  `sha256: null` for every file larger than 512 MiB. The curated-copy rules then
  exclude checkpoints, exported weights, TensorBoard files, and per-step reward
  Parquets (`:60-72`), while the archive checksum covers only the bundle that
  omits them (`:121-126`). The export audit records tensor shapes/counts but no
  weight digest (`scripts/evaluation/export_fve_checkpoint.py:59-97,168-179`).
- **Impact:** the final result cannot identify or later verify the exact large
  checkpoint/model artifacts from which Rotation predictions were produced.
  Different same-shaped weights with the same pathname and byte size generate
  indistinguishable inventory entries.
- **Reproduction/static test:** direct inspection shows the only hash branch is
  `size <= 512 * 1024 * 1024`; Qwen weight/checkpoint shards above that threshold
  necessarily receive `null`. No chunk hash, tree hash, or external checksum is
  recorded elsewhere in the packaging chain.
- **Recommended fix:** stream-hash every omitted regular file regardless of
  size, or record a deterministic chunked/Merkle digest plus total size. Bind
  the exported-model digest set into post-training, benchmark, table, and final
  manifests so lineage is checkable without embedding the weights.

## Areas checked with no additional confirmed issue

- Stage subprocess output is kept under the durable run root, queue logs append
  rather than truncate, and Ray cleanup is attempted in a `finally` block after
  every training stage (`run_queue.py:18-37,85-105`).
- Reward summaries use a sibling temporary file plus `os.replace` and refuse an
  empty trace directory (`summarize_rewards.py:17-43`).
- Evaluation-data downloads use a temporary file, verify the pinned MD5, and
  replace the target only after validation (`prepare_evaluation_data.py:57-92`).
- New shard-only checkpoint exports are built and audited in a temporary sibling
  directory before rename (`export_fve_checkpoint.py:142-165`); the confirmed
  reuse/provenance gap concerns already-existing targets and checkpoint choice.
- Table aggregation loads all inputs before writing outputs and rejects missing
  score keys/files. Its multi-file output is not transactionally staged, but no
  silent wrong-result path beyond the upstream stale/provenance findings was
  demonstrated.
- No focused repository tests cover `run_queue.py`, checkpoint resolution,
  post-training state, six-benchmark reuse, table publication, or final-package
  failure recovery.

## Commands and checks run

- Fully read root `AGENTS.md` and inspected `EVALUATION.md` and
  `FROZEN_VISION_TRAINING.md` for the required durable workflow.
- Inventoried and traced relevant files with `rg --files`, `rg -n`, `find`,
  `git log`, `git status --short`, `wc -l`, `nl -ba`, and `sed`.
- Read the queue/watcher/runtime/reward-summary paths, checkpoint trainer and
  resolver, post-training smoke, evaluation-data and six-benchmark manifests,
  table aggregation, and final packager.
- Ran a temporary checkpoint fixture proving tracker step 1 resolves to partial
  step 2; all files were automatically removed.
- Ran a fake CPU-only pinned-evaluator fixture proving six old score artifacts
  are accepted for a newly claimed model and bogus dataset digests are ignored;
  all files were automatically removed.
- Ran the real packager on temporary mismatched/invalid manifests and inspected
  its tar members to confirm missing gate/data/state evidence; all files were
  automatically removed.
- Injected a temporary tar-add failure and retried to confirm the partial-file
  dead end; all files were automatically removed.
- Ran the real queue against a temporary zero-byte completion marker to confirm
  the presence-only false-success path; all files were automatically removed.
