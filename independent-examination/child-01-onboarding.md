# Independent examination: environment, model, and data onboarding

Scope was limited to static inspection and safe synthetic/unit checks. No network, SSH, GPU, model inference, dataset download, or smoke run was used.

## Confirmed issues

### 1. High: the pinned base-model identity is not enforced before training

- **Evidence:** `AGENTS.md:181-193` pins revision `66285546...` but validates only `c.model_type == 'qwen2_5_vl'`; the stated two-shard/7,509,245,952-byte expectation is not checked. `scripts/frozen_vision/run_frozen_vision.py:35-53,134-135` resolves and forwards any supplied path. The claimed authoritative runtime audit in fact only checks that a `visual` module exists, freezes it, and prints counts (`verl/workers/fsdp_workers.py:330-345`); it never compares the frozen, trainable, or total counts to the required values.
- **Impact:** another Qwen2.5-VL size, a locally modified checkpoint, or a same-type checkpoint at a different revision can pass onboarding and start an expensive run. The resulting experiment would not be attributable to the pinned base model even if all freeze checks pass.
- **Reproduction/static test:** inspection found no comparison with `668,684,288`, `3,085,938,688`, `3,754,622,976`, the two expected shard names, their aggregate tensor bytes, or model-file hashes anywhere in the FVE launch path. A config for any checkpoint with `model_type=qwen2_5_vl` satisfies the documented preflight command.
- **Recommended fix:** add a CPU-only `validate_base_model.py` (and call it from `run_queue.py` before creating run state) that requires the exact architecture/config, expected shard set, safetensors tensor-byte total and parameter counts, and a durable revision/hash manifest. Fail closed before any Ray/GPU launch; change the worker's count log into exact assertions for the supported base model.

### 2. High: `--verify-only` can approve self-attested or schema-incompatible Rotation data

- **Evidence:** `scripts/frozen_vision/prepare_rotation_data.py:117-135` checks column names, row counts, only the first `data_source`, and a computed hash. The reference output hashes are recorded as informational `matches_reference_247` booleans and are not required. `validate_existing` checks only source name/revision and equality with hashes stored by that same local manifest (`:139-158`); it ignores `complete`, seed, granularity, split mapping, raw hashes, package versions, and generator commit. The manifest stores column names but not Arrow field types (`:128-135,365-379`), despite `AGENTS.md:216-220` promising a matching schema/provenance manifest.
- **Impact:** a full-row-count Parquet with wrong nested types or wrong Rotation contents can be paired with its own hashes and pass `--verify-only`. This yields either a late training-loader failure or, worse, a silent run on non-canonical data.
- **Reproduction/static test:** a temporary manifest with the pinned source/revision but `seed=999`, `granularity_degrees=1`, `complete=false`, and a bogus split mapping was accepted when the file validator returned hashes matching the manifest (`MISMATCHED_BUILD_METADATA_ACCEPTED=True`). The source also shows that even a reference-hash mismatch merely sets a false field.
- **Recommended fix:** define and compare an exact Arrow schema (including nested `images`, `prompt`, `reward_model`, and `extra_info` types); validate every row's task/source and allowed answers; require all build-invariant manifest fields and raw hashes; and, for complete builds under the pinned package family, require the canonical generated hashes or a separately signed/trusted provenance record rather than a self-declared hash.

### 3. Medium: the mandatory environment gate does not enforce the validated package family

- **Evidence:** `AGENTS.md:21-45` specifies exact Python and package versions, but gate 1 at `AGENTS.md:106-110` only prints versions and asserts that Torch advertises `sm_90`. It does not assert Python 3.10, Torch/CUDA 2.8.0+cu128, or any listed package version, and it omits several pinned components. In addition, plain `pip install -e .` is prescribed (`AGENTS.md:87-94`) while `setup.py:26-44` leaves packages such as pandas, wandb, dill, codetiming, and pylatexenc unconstrained and has no `python_requires` guard.
- **Impact:** a drifted or partially mixed environment can pass all documented version checks. Fresh builds are not fully reproducible because resolution of the remaining direct/transitive dependencies changes over time; later CUDA or training failures can therefore be incorrectly attributed to the model code.
- **Reproduction/static test:** AST/text inspection of the gate found one assertion total, for `'sm_90'`; changing any printed version would not change its exit status. The editable install necessarily resolves the unpinned requirements at install time.
- **Recommended fix:** ship a Python-3.10 constraints/lock file for the entire validated environment (including transitive versions/hashes where practical), install editable code under that constraint, declare `python_requires`, and replace the print-only gate with exact normalized-version assertions for every compatibility-critical package and CUDA build.

### 4. Medium: smoke-data preparation silently accepts undersized input and emits an invalid production-shape gate

- **Evidence:** `scripts/frozen_vision/prepare_smoke_data.py:13-19` reads at most one batch, slices it, and writes whatever row count is available without checking it; `:31-32` requests 10 and 512 rows. No source manifest/hash/schema is validated and no smoke manifest is written. `run_frozen_vision.py:81-107` then treats these files as the quick and production-shape inputs without a preflight row-count check.
- **Impact:** a truncated/stale source can produce a `production_shape.parquet` with fewer than 512 rows while the command exits successfully. The purported production-shape memory/batch gate can then fail late or exercise the wrong shape, weakening the guard before the long run.
- **Reproduction/static test:** a stubbed Parquet source yielding 3 rows was passed to `write_prefix(..., rows=512)`; it printed success and recorded `REQUESTED_ROWS=512 WRITTEN_ROWS=3`. AST inspection found no `if`, `assert`, or raise guard in `write_prefix`.
- **Recommended fix:** validate the canonical Rotation manifest and exact Arrow schema first; require `source_rows >= requested_rows` and `len(table) == requested_rows`; write atomically; and emit a smoke manifest containing source data hash, selected row indices, output hashes/schema, and exact row counts. Recheck these fields in the queue preflight.

### 5. Medium: data-root defaults disagree across the onboarding commands

- **Evidence:** `prepare_rotation_data.py:282-311` defaults generated data to `<output-root>/inputs/datasets`, while `prepare_smoke_data.py:23-29`, `run_frozen_vision.py:33-46`, `run_queue.py:43-56`, and `gpu_watcher.py:57-69` default the data root to checkout-local `our_datasets` when `FVE_DATA_ROOT` is absent.
- **Impact:** a valid sequence that consistently supplies only `--output-root` creates Rotation data successfully and then fails to find it during smoke preparation or launch. It can also accidentally select stale checkout-local data instead of the durable dataset.
- **Reproduction/static test:** static path evaluation gives `<O>/inputs/datasets/MMBench_RotationQA_90degree` for preparation but `<repo>/our_datasets/MMBench_RotationQA_90degree` for every downstream consumer. The documented full setup avoids this only because it explicitly exports `FVE_DATA_ROOT`.
- **Recommended fix:** centralize the data-root default in `fve_runtime.py` as `<resolved output-root>/inputs/datasets` and use it in all four commands; keep an explicit `--data-root` override and print the resolved value in every preflight.

## Areas checked with no confirmed issue

- The raw MMBench snapshot is pinned and both the exact raw file set and SHA-256 values are enforced (`prepare_rotation_data.py:161-181`).
- Partial development builds use a suffixed output directory and cannot overwrite the canonical dataset (`prepare_rotation_data.py:293-312`).
- Rotation outputs are built in a temporary sibling directory and renamed only after validation/manifest creation (`prepare_rotation_data.py:326-381`).
- The 8-GPU train dry run rendered rollout batch 2560, PPO mini-batch 640, per-GPU mini-batch 80, and micro-batch 10. A temporary-path check found no dry-run-created artifact; the only observed file was the audit's redirected stdout, which was removed.
- GPU gate 2 performs a real operation on all eight visible GPUs, gate 3 invokes a real FlashAttention kernel, and gate 5 resolves the Ray CLI beside the selected Python. These were inspected only, not executed.
- Syntax compilation succeeded for `fve_runtime.py`, both data-preparation scripts, `run_frozen_vision.py`, and `run_queue.py`, with bytecode redirected outside the checkout and deleted.

## Commands/checks run

- Fully read `AGENTS.md` (`wc -l`; `sed -n '1,240p'`; `sed -n '241,430p'`).
- Inventoried and traced relevant files with `rg --files`, `rg -n`, `nl -ba`, `sed`, `find`, and `git status --short`.
- Ran a temporary synthetic `validate_existing` manifest-mismatch unit check; all temporary files were automatically removed.
- Ran a stubbed three-row `write_prefix(..., 512)` unit check; its temporary directory was automatically removed.
- Ran the 8-GPU train `--dry-run` against temporary model/data/output paths, enumerated artifacts, and removed the temporary directory.
- Ran `python3 -m py_compile` with `PYTHONPYCACHEPREFIX` set to a temporary directory, then removed it.
- Attempted a real PyArrow synthetic Parquet check, but the local audit interpreter has no PyArrow (nor Torch, Transformers, or datasets). Per scope, no packages or data were downloaded; the check was replaced by bounded stubs/static inspection and the missing dependency was not treated as a repository defect.
