# Independent examination: evaluation and Tables I--III

## Conclusion

The full six-benchmark/Table I--III chain is **not yet reportable**. The table
mean and category projections are arithmetically sound on synthetic inputs, but
the current runner can reuse stale predictions under a new model path, does not
authenticate the datasets or evaluator it actually executes, and uses an
incorrect generic MCQ class for RealWorldQA. The held-out 60/40 detail protocol
is not implemented: the aggregator accepts arbitrary, unpaired CSV overrides.

No network, SSH, GPU, inference, dataset download, or smoke run was used. All
dynamic checks below used temporary CPU-only synthetic fixtures.

## Findings

### 1. Critical: RealWorldQA is configured as a multiple-choice benchmark

**Evidence.** `scripts/evaluation/run_six_benchmarks.py:15-22` assigns
`ImageMCQDataset` to every benchmark except MME-RealWorld-Lite, including
RealWorldQA at line 19. The same runner globally requests `exact_matching` at
lines 109-119. RealWorldQA is an open-answer VQA dataset (question/answer, not
A--D options); the pinned VLMEvalKit protocol requires its VQA dataset class
and dataset-specific evaluation path.

**Impact.** The full run can either fail while constructing/evaluating
RealWorldQA or prompt and score it as the wrong task. A resulting Table I
RealWorldQA number would not be protocol-comparable.

**Static reproduction.** Inspect the emitted dry-run config: the RealWorldQA
entry is exactly `{"class": "ImageMCQDataset", "dataset": "RealWorldQA"}`.
No schema check in `prepare_evaluation_data.py:95-143` can catch the absence of
MCQ option columns.

**Recommended fix.** Use the pinned commit's RealWorldQA-specific supported
class (normally `ImageVQADataset`) and official evaluator. Add a CPU fixture
with the real TSV header and one synthetic row that verifies prompt construction,
answer normalization, expected score-artifact name, and scale.

### 2. High: the implemented MCQ prompt and image preprocessing do not reproduce the repository's paper-evaluation recipe

**Evidence.** `README.md:200-204` requires a customized `image_mcq.py` prompt
whose final answer is `<think>...</think> <answer>$LETTER</answer>`. The setup
only applies the Qwen model-directory detection patch
(`scripts/evaluation/setup_vlmevalkit.sh:27-29`; the patch contents are solely
in `scripts/evaluation/patch_vlmeval_qwen.py:28-50`). The run config sets
`use_custom_prompt=False` at `run_six_benchmarks.py:102`, so the required MCQ
prompt is never installed.

The paper-evaluation example specifies `min_pixels=35840` at
`README.md:210-218`, while the full runner uses `1280 * 28 * 28 = 1,003,520`
at `run_six_benchmarks.py:93-103`; the smoke repeats it at
`vlmeval_mmbench_smoke.py:166-176`. This is a 28x larger minimum pixel budget.

**Impact.** Prompting, answer extraction, visual token count, latency, memory,
and accuracy can differ from the stated paper recipe. Applying the same drift
to Base and Rotation makes the comparison symmetric but does not make it a
paper reproduction.

**Static reproduction.** `python3 -c 'print(1280*28*28, 35840,
(1280*28*28)/35840)'` prints `1003520 35840 28.0`; searching the setup and
patch scripts finds no `image_mcq.py` modification.

**Recommended fix.** Resolve the authoritative prompt and pixel parameters
against the actual paper run, encode them in a checked patch/config, and record
their content hashes in each run manifest. Add a test that asserts the rendered
prompt and all generation/image parameters, not only the JSON constants.

### 3. Critical: `--reuse` can silently relabel old predictions and scores as a new checkpoint run

**Evidence.** Output identity is only `<run-id>/<model-name>`
(`run_six_benchmarks.py:88-89`), while the evaluator is always invoked with
`--reuse` (`run_six_benchmarks.py:109-119`). `require_scores` merely globs for
one filename per benchmark and never checks content, timestamps, prediction
files, sample counts, model identity, or hashes (`run_six_benchmarks.py:32-43`).
The subsequently written manifest records the *current* `model_path` at lines
132-143 even if reuse consumed artifacts made by an older path.

**Impact.** Re-running the same run ID/model label after changing or re-exporting
a checkpoint can produce a manifest that names the new model but contains the
old model's predictions/scores. This can invert or erase reported gains without
an error.

**CPU fixture.** Six correctly named files with mtime `1` and arbitrary text
were placed under a synthetic `work_dir/Base`; `require_scores` accepted all
six (`stale_scores_accepted=6`, oldest mtime `1.0`).

**Recommended fix.** Make each attempt immutable and fingerprinted by model
weights/export manifest, dataset hashes, evaluator patch hash, and config hash.
Refuse a non-empty result directory by default. If resume is needed, validate
those fingerprints before reuse and verify prediction row/index coverage plus
score schemas before declaring completion.

### 4. High: runner preflight trusts mutable paths and a self-asserted evaluator marker instead of the pinned bytes

**Evidence.** Data preparation contains fixed URLs and MD5 values
(`prepare_evaluation_data.py:17-46`) and can verify them, but the actual runner
only checks that each path named by the JSON manifest exists
(`run_six_benchmarks.py:75-86`). It neither recomputes a checksum nor requires
that the manifest path equals the `$LMUData/<dataset>.tsv` file that VLMEvalKit
will load. Partial preparation also merges untouched old manifest entries
(`prepare_evaluation_data.py:125-142`).

Evaluator validation is only the text of `.pinned-commit`
(`run_six_benchmarks.py:67-70`); it does not verify Git HEAD, dirty state, or
the expected compatibility patch. The run manifest stores a mutable data-
manifest pathname and a hard-coded commit string, but no dataset, evaluator,
config, model, prediction, or score hashes (`run_six_benchmarks.py:132-147`).
The environment setup additionally uses `--system-site-packages` and resolves
mostly unpinned requirements (`setup_vlmevalkit.sh:31-45`), while the run
manifest records no package versions.

**Impact.** Corrupted/substituted data or arbitrarily edited evaluator code can
be reported as the pinned six-benchmark protocol. Later mutation of the data
manifest destroys the evidentiary link for an already completed run.

**CPU fixture.** A temporary fake evaluator containing only an executable
Python symlink, a no-op `run.py`, and the expected marker, plus six corrupted
`$LMUData` TSVs and a manifest pointing to unrelated decoy files, passed the
runner's dry-run preflight with return code 0.

**Recommended fix.** At run start, require canonical dataset paths and recompute
all expected MD5/SHA256 values; validate basic schema and expected row counts.
Verify Git HEAD and an allowlisted dirty diff/patch digest. Snapshot the data
manifest into the result directory and record hashes for it, config, evaluator,
model/export manifest, predictions, scores, and package versions.

### 5. High: the V* dataset/protocol identity is not source-grounded inside the repository

**Evidence.** `EVALUATION.md:21-25` declares Table I's V* to be standard
`VStarBench` with no crop/tool loop, and the runner hard-codes that choice at
`run_six_benchmarks.py:15-22`. However, the repository's own list of
"benchmarks used in the paper" lists MMStar but not VStar
(`README.md:131-140`). Immediately before commit `757dc67`, the tracked
evaluation specification instead called for "V*Bench with the agentic
visual-search protocol" and a separate runner (`git show
757dc67^:EVALUATION.md`, old lines 8-13 and 115-122). No paper table, protocol
citation, or provenance record in the current checkout resolves these three
different interpretations.

**Impact.** A complete, technically successful run may still fill the V* column
with the wrong benchmark or protocol, making Table I non-reproducible and
non-comparable to the target table.

**Static reproduction.** `rg -n -i 'VStar|MMStar|V\\*Bench'` over the current
checkout plus the `git show` command above exposes the conflict; there is no
evaluation implementation other than generic `ImageMCQDataset`.

**Recommended fix.** Bind the column to a cited paper table/dataset release and
explicit protocol. If standard VStarBench is correct, add the authoritative
evidence and expected split/count/category checks. If the agentic protocol is
correct, restore a separately pinned runner rather than substituting generic
single-pass MCQ evaluation.

### 6. High: the claimed 60/40 held-out detail protocol is absent and arbitrary/unpaired overrides are accepted

**Evidence.** `EVALUATION.md:150-152` says Tables II/III require separate 60/40
held-out runs. No script constructs or records that split. The aggregator's four
override arguments are independently optional (`aggregate_paper_tables.py:135-146`),
and `load_run` accepts each raw CSV solely by pathname (`aggregate_paper_tables.py:69-93`).
It does not require all four overrides together, validate dataset/model/split
identity, compare sample indices, or compare base and rotation run IDs,
VLMEvalKit commits, data manifests, or generation configs. The output records
only override paths, not hashes or split metadata (`aggregate_paper_tables.py:160-170`).

**Impact.** Tables II/III can mix full-set and held-out results, use different
held-out examples for Base and Rotation, swap model files, or combine unrelated
runs while still producing polished CSV/JSON/LaTeX. Such differences cannot be
interpreted as model improvements.

**CPU fixture.** Synthetic manifests with model labels `NotBase`/`NotRotation`,
run IDs `run-A`/`run-B`, and `vlmeval_commit=WRONG` were accepted. Supplying only
`--base-mmbench-detail` also succeeded: Base Table II became `90.00`, while its
Table I MMBench remained `50.00`; the other three overrides stayed null.

**Recommended fix.** Implement a deterministic held-out split generator that
records source checksum, split seed/method, exact indices, and split checksum.
Consume detail *run manifests*, not naked CSVs. Enforce either zero overrides
or a complete compatible set of four, with identical split IDs and protocol
fingerprints across models. Reject mismatched run/protocol/model roles.

### 7. Medium: score-unit inference is discontinuous and score values are not validated

**Evidence.** `aggregate_paper_tables.py:39-45` assumes every value with
absolute magnitude at most `1.0000001` is a fraction and multiplies it by 100;
larger values are assumed already-percent. There are no dataset-specific unit,
finite-value, or range checks in `read_score_csv`/`read_mme`
(`aggregate_paper_tables.py:54-66`).

**Impact.** An official percentage value of exactly `1.0` becomes `100.0`, while
`1.01` stays `1.01`. NaN/Infinity or out-of-range values can also enter table
artifacts. Mixed evaluator schemas cannot be safely inferred from magnitude.

**CPU fixture.** Direct calls produced `percent(0.5)=50.000000`,
`percent(1)=100.000000`, and `percent(1.01)=1.010000`.

**Recommended fix.** Define the expected unit and schema per benchmark/artifact,
validate finite ranges and required columns, and convert explicitly. Include
representative official-format fixtures for all six datasets.

## Checked areas that passed

- The configured benchmark names nominally match the current `AGENTS.md` scope:
  MMBench DEV-EN v1.1, SEEDBench_IMG, VStarBench, RealWorldQA, BLINK, and
  MME-RealWorld-Lite. Fixed download checksums exist for all six, and the data
  preparation path verifies downloaded/reused bytes when it is actually run.
- Base and Rotation use the same runner constants: temperature `0.01`, top-p
  `0.001`, top-k `1`, max tokens `2048`, and the same pixel settings.
- Table I computes an unweighted arithmetic mean of exactly six scores
  (`aggregate_paper_tables.py:81-90`). A synthetic `50,60,70,80,90,40` vector
  produced `65`.
- MMBench and SEED column maps contain the requested six and nine detail
  categories, respectively (`aggregate_paper_tables.py:14-36`), and their
  `Overall` values are kept as the weighted evaluator overall rather than
  incorrectly re-averaging categories.
- Decimal conversion occurs before cross-benchmark averaging, and half-even
  table rounding is intentional: `53.365 -> 53.36`.
- All eight Python files under `scripts/evaluation` parsed successfully with
  `ast.parse`. No dedicated tests for data preparation, the six-benchmark
  runner, score discovery/parsing, or table aggregation were found under
  `tests/`.

## Commands/checks used

- `sed -n '1,520p' AGENTS.md` (read completely before audit)
- `nl -ba EVALUATION.md` and `nl -ba scripts/evaluation/{prepare_evaluation_data,run_six_benchmarks,aggregate_paper_tables,vlmeval_mmbench_smoke}.py`
- `nl -ba scripts/evaluation/{setup_vlmevalkit.sh,patch_vlmeval_qwen.py,package_final_results.py}`
- `rg -n 'MMBench_DEV_EN_V11|SEEDBench_IMG|VStarBench|RealWorldQA|BLINK|MME-RealWorld-Lite|held-out|60/40' .`
- `git show 757dc67^:EVALUATION.md` and relevant `git log -S...` checks
- CPU temporary-fixture test of runner preflight, marker validation,
  `require_scores`, mismatched manifests, and partial detail overrides
- CPU direct checks of percentage conversion, six-score mean, and rounding
- `ast.parse` of all evaluation Python files; `rg` for dedicated test coverage

Remote checksum contents and the pinned VLMEvalKit implementation were not
independently fetched because this examination explicitly prohibited network
access and dataset downloads. That limitation makes Findings 1 and 5 mandatory
pre-run protocol gates rather than matters to defer until after expensive
evaluation.
