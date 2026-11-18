# Reward-difficulty analysis closure

Date: 2026-11-18

## Reviewer-facing operational definition

The primary quantity is the fraction of prompt groups whose rollout rewards
have non-zero within-group population variance. This is directly connected to
the GRPO implementation: when all rewards in a `uid` group are identical,
subtracting the group mean yields zero reward-driven advantage for that group.

The analysis explicitly distinguishes this from global reward standard
deviation. A CPU contract verifies the counterexample in which half the prompt
groups are all-zero and half are all-one: global reward standard deviation is
0.5, but informative-group fraction and mean within-group variance are both
zero.

## Implementation

- `verl/utils/reward_difficulty.py` provides shared group and aggregate metric
  definitions.
- `RewardTraceRecorder` emits the group-aware metrics to the training logger.
- `analyze_reward_difficulty.py` validates every trace group and produces
  prompt-level Parquet, step/stage CSVs, trends, summary, and an immutable
  manifest.
- `run_queue.py` runs the analysis after every stage and binds its manifest into
  `queue.complete`.
- `package_final_results.py` revalidates the manifest and derived file hashes,
  then includes the compact derived evidence without duplicating raw traces.
- Legacy queues can run the analyzer post hoc and provide the manifest through
  `--reward-difficulty-manifest`.

## End-to-end synthetic validation

An isolated temporary environment using the pinned NumPy 1.26.4 and PyArrow
20.0.0 generated three trace steps with five rollouts per prompt. Expected and
observed informative-group fractions were:

```text
step 1: 0.0
step 2: 0.5
step 3: 1.0
```

The analyzer published all five expected artifacts, matched checkpoint step 3,
and a second invocation validated and reused the immutable outputs. The
dependency-light workflow suite also passed 25 tests before this end-to-end
fixture.

## Scientific boundary

The current Rotation run can establish reward-signal dynamics over training.
Testing whether this quantity predicts downstream improvement across SSL task
difficulty requires the same traces for the compared Rotation, Jigsaw,
Position, Contrastive, and Mask Identification conditions. Historical runs
without per-rollout traces cannot be reconstructed after the fact.
