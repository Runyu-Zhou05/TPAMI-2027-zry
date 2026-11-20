# Evaluation artifact audit (2026-11-20)

Archive: `artifacts/1120/fve_eval_package_20261120.tar.gz`

## Confirmed integrity

- The archive has 215 safe members and no path traversal or link entries.
- Rotation is the committed `global_step_260` direct HF checkpoint with
  3,754,622,976 total and 668,684,288 frozen visual parameters.
- Dataset MD5/SHA256 identities and exact-matching denominators agree with the
  score artifacts.
- Base is the pinned Qwen2.5-VL-3B snapshot; Rotation weights are bound to the
  checkpoint/export manifests.

## Protocol defects found

1. The generic SSL4RL MCQ prompt patch was not effective. Across Base MMBench,
   Base RealWorldQA, and all applicable Rotation datasets, prediction coverage
   for both `<think>` and `<answer>` tags is exactly 0%. Predictions use direct
   forms such as `A. option text`.
2. The patcher's former idempotence marker was the generic phrase "The last
   line of your response should be of the following format:". That phrase
   already exists in the pinned source's unrelated MMMU-Pro COT branch, so a
   pristine file was incorrectly reported as already patched.
3. The paper-literal Rotation manifest records `min_pixels=35840` for all six
   datasets, but the run log uses `--reuse` and the package README states that
   only MMBench was regenerated; the other five prediction sets came from the
   earlier default-resolution attempt. The manifest therefore combines two
   effective protocols.
4. The token truncation experiment cannot diagnose the paper's COT token budget
   because the evaluated prompts produced answer-only outputs. Its zero score
   change is valid for these stored predictions but not for the intended
   `<think>...</think><answer>...</answer>` protocol.

## Behavioral evidence

- Base MMBench prediction mean length: 26.6 characters; tag coverage: 0%.
- Base RealWorldQA mean length: 19.0 characters; tag coverage: 0%.
- Rotation MMBench mean length: 33.5 characters; tag coverage: 0%.
- Rotation SEED mean length: 15.6 characters; tag coverage: 0%.
- Base and Rotation RealWorldQA predictions are exactly identical for 738 of
  765 samples (96.47%). Rotation changes only 27 predictions, with 15
  wrong-to-correct and 7 correct-to-wrong transitions.

These artifacts are internally valid for the direct-answer protocol but are
not a faithful execution of the paper README's customized COT prompt.
