#!/usr/bin/env python3
"""Exercise the Qwen2.5-VL monkey-patch path used by FVE training."""

from __future__ import annotations

import importlib.metadata
import sys
import types

from packaging.version import Version


def validate_transformers_patch() -> dict:
    transformers_version = importlib.metadata.version("transformers")
    if Version(transformers_version) < Version("4.55.0"):
        raise RuntimeError(f"This FVE environment expects Transformers >=4.55, found {transformers_version}")

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.models.transformers.qwen2_5_vl import forward_with_torch_backend, forward_with_triton_backend
    from verl.models.transformers.qwen2_vl import get_rope_index

    module_name = "_fve_qwen25_transformers_compat_probe"
    fake_module = types.ModuleType(module_name)
    sys.modules[module_name] = fake_module

    class FakeConfig:
        model_type = "qwen2_5_vl"
        num_attention_heads = 16
        num_key_value_heads = 2

    class FakeQwenModule:
        __module__ = module_name
        config = FakeConfig()

    try:
        apply_monkey_patch(
            FakeQwenModule(),
            ulysses_sp_size=1,
            use_remove_padding=True,
            use_fused_kernels=False,
        )
    finally:
        sys.modules.pop(module_name, None)

    return {
        "apply_monkey_patch": "ok",
        "symbols": [
            get_rope_index.__name__,
            forward_with_torch_backend.__name__,
            forward_with_triton_backend.__name__,
        ],
        "transformers": transformers_version,
    }


def main() -> None:
    print(validate_transformers_patch())


if __name__ == "__main__":
    main()
