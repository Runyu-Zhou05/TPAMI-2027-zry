# Operational reward difficulty for GRPO

## Primary quantity

For prompt group `g` with rollout rewards `r_g1, ..., r_gK`, define an
informative group as:

```text
Var(r_g1, ..., r_gK) > epsilon
```

The primary difficulty-calibration statistic at training step `t` is:

```text
informative_group_fraction(t)
  = (# prompt groups with non-zero within-group reward variance)
    / (# prompt groups)
```

This quantity is directly tied to standard GRPO. If every rollout in a prompt
group has the same outcome reward, subtracting the group mean produces zero
reward-driven advantage for every rollout in that group. The default numerical
tolerance is `epsilon = 1e-12`.

Global reward standard deviation is not a substitute. A batch containing some
all-zero prompt groups and some all-one prompt groups has non-zero global
standard deviation while every prompt group has zero within-group variance and
therefore no reward-driven GRPO signal.

## Secondary quantities

For binary verifiable rewards, the analysis additionally reports:

- all-fail group fraction;
- all-success group fraction;
- mixed-success group fraction;
- rollout success rate;
- mean within-group reward variance;
- mean Bernoulli reward entropy in bits.

For non-binary rewards, entropy is the empirical categorical entropy after
rounding rewards to eight decimal places. The rounding precision, success
threshold, and variance epsilon are recorded in the analysis manifest.

## Durable outputs

Every production step already writes one row per rollout under:

```text
<run-dir>/reward_traces/rotation/step_XXXXXXXX.parquet
```

`analyze_reward_difficulty.py` validates prompt identities, rollout IDs, group
sizes, run/task identity, contiguous steps, finite rewards, and final checkpoint
agreement. It publishes:

```text
reward_difficulty_groups.parquet
reward_difficulty_by_step.csv
reward_difficulty_by_stage.csv
reward_difficulty_summary.json
difficulty_analysis_manifest.json
```

The stage CSV partitions observed steps into equal-count early, middle, and
late segments and also reports an overall row. The JSON includes normalized-
progress AUC and linear trends for informative-group fraction, reward entropy,
and success rate.

## Manual analysis of an existing Rotation run

```bash
python scripts/frozen_vision/analyze_reward_difficulty.py \
  "$FVE_OUTPUT_ROOT/runs/<run-id>/ssl4rl_qwen_3b_mmbench_rotation_frozen_vision/reward_traces/rotation" \
  --output-dir "$FVE_OUTPUT_ROOT/runs/<run-id>/ssl4rl_qwen_3b_mmbench_rotation_frozen_vision/reward_analysis/rotation/train" \
  --expected-run-id <run-id> \
  --expected-task rotation \
  --expected-rollouts-per-prompt 5 \
  --checkpoint-root "$FVE_OUTPUT_ROOT/runs/<run-id>/ssl4rl_qwen_3b_mmbench_rotation_frozen_vision/checkpoints"
```

New guarded queue runs execute this automatically after each quick,
quick-resume, production-smoke, and train stage. Final result packaging includes
the derived group Parquet, step/stage CSVs, summary, and analysis manifest, but
does not duplicate the much larger raw per-response trace Parquets.

For a queue that completed before automatic analysis was added, pass the
generated train manifest explicitly when packaging:

```text
--reward-difficulty-manifest \
  <run-dir>/reward_analysis/rotation/train/difficulty_analysis_manifest.json
```

## Cross-variant predictive analysis

One Rotation run is sufficient to characterize reward-signal dynamics over
training, but it is not sufficient to establish a predictive relationship
between difficulty calibration and downstream improvement. That analysis
requires the same trace and metric definitions for each compared condition,
such as Rotation 90/45 degrees, Jigsaw 2x2/3x3, Position 2x2/3x3, Contrastive
augmentation strengths, and Mask Identification model scales. A suitable
variant-level predictor is the AUC of `informative_group_fraction(t)` or its
early/middle/late values, correlated with the corresponding final downstream
gain using Spearman correlation.
