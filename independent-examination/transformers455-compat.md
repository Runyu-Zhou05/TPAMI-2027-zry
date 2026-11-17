# Transformers 4.55 compatibility closure

Date: 2026-11-17

## Finding

Restoring the original SSL4RL `verl/models/` package fixed the missing-module
failure, but its Qwen monkey patch deliberately raised
`ValueError: Transformers 4.53 is not supported` for every Transformers version
at or above 4.53. This contradicted the validated environment pin
`transformers==4.55.2` in `AGENTS.md`.

## Fix

The following Qwen transformer adapters were updated from upstream verl commit
`9b6a07fa77654372c457bd16dc89fe3ab10310ed` (2025-08-25), whose stated stack is
vLLM 0.10, mcore 0.13, and Transformers 4.55:

- `verl/models/transformers/monkey_patch.py`
- `verl/models/transformers/qwen2_vl.py`
- `verl/models/transformers/qwen2_5_vl.py`

The clean repository also needed `verl/utils/transformers_compat.py`, an
upstream compatibility dependency that happened to exist in the colleague's
locally modified checkout but was absent from a fresh clone.

The new code selects `Qwen2_5_VLAttention` for the post-4.53 API, handles the
Transformers 4.55 flash-attention `query_length` signature, uses the newer
Qwen2.5-VL base-model forward API, and normalizes attention return arity.

## Prevention and runtime evidence

`check_environment.py` now does more than import the worker: it calls
`apply_monkey_patch` on a Qwen2.5-VL-shaped probe under the installed
Transformers package. The target-side reproduction reported:

```text
Monkey patch FlashAttention2.forward in Qwen2.5VL
Monkey patch _flash_attention_forward in fake_qwen_module for new api
apply_monkey_patch OK (transformers 4.55 path)
```

The environment gate also rejects effective pod/cgroup memory below 160 GiB.
The job specification should request at least 180 GB because the 64 GB attempt
OOMed with parameter and optimizer offload enabled.
