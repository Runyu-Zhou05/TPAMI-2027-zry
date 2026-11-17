# Recovery of the accidentally omitted `verl.models` package

Date: 2026-11-17

## Root cause

The FVE artifact section of `.gitignore` used the unanchored pattern
`models/`. Git applies that pattern to a matching directory at any depth, so
the required upstream Python package `verl/models/` was ignored together with
the intended repository-root generated model directory. Consequently, the
GitHub checkout could pass shallow tests while failing as soon as
`verl/workers/fsdp_workers.py` imported
`verl.models.transformers.monkey_patch`.

## Recovery source and compatibility

The complete source package was recovered from the original neighboring
`SSL4RL-main/verl/models/` checkout rather than from a moving upstream branch.
Both trees report bundled verl version `0.4.1.dev`. A recursive comparison of
all other `verl/` files found equality except the seven files intentionally
changed by this FVE project (checkpoint/export, trainer config, reward trace,
and FSDP worker patches). The recovered directory contains 60 files, including
58 Python modules and the Qwen2.5-VL transformer/mcore implementations used by
the current worker code.

## Prevention

- `.gitignore` now uses `/models/`, which ignores only generated artifacts at
  the repository root and leaves `verl/models/` trackable.
- A dependency-free source-tree contract resolves every `verl.models...`
  import in the repository to a shipped module.
- The target environment gate and `tests/special_sanity/test_import.py` now
  import the real FSDP production worker, not merely the top-level `verl`
  package.
