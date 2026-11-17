#!/usr/bin/env python3
"""Fail closed unless the selected interpreter matches the validated FVE stack."""

from __future__ import annotations

import importlib.metadata
import json
import sys


REQUIRED = {
    "accelerate": "1.8.1",
    "datasets": "4.0.0",
    "flash_attn": "2.8.3.post2",
    "hydra-core": "1.3.8",
    "numpy": "1.26.4",
    "omegaconf": "2.3.1",
    "peft": "0.21.2",
    "pyarrow": "20.0.0",
    "qwen-vl-utils": "0.0.11",
    "ray": "2.60.0",
    "tensordict": "0.9.1",
    "tokenizers": "0.21.1",
    "torch": "2.8.0+cu128",
    "torchaudio": "2.8.0+cu128",
    "torchdata": "0.11.0",
    "torchvision": "0.23.0+cu128",
    "transformers": "4.55.2",
    "triton": "3.4.0",
    "vllm": "0.10.2",
    "xformers": "0.0.32.post1",
}


def validate_environment() -> dict:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"Python 3.10 is required, found {sys.version.split()[0]}")
    installed = {}
    mismatches = []
    for distribution, expected in REQUIRED.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[distribution] = actual
        if actual != expected:
            mismatches.append(f"{distribution}: expected {expected}, found {actual}")
    if mismatches:
        raise SystemExit("Validated FVE dependency mismatch:\n" + "\n".join(mismatches))

    import torch

    if torch.version.cuda != "12.8":
        raise SystemExit(f"Torch must use CUDA 12.8, found {torch.version.cuda}")
    architectures = torch.cuda.get_arch_list()
    if "sm_90" not in architectures:
        raise SystemExit(f"Torch lacks H20 sm_90 kernels: {architectures}")

    # Import the actual production worker and the source package that was once
    # accidentally hidden by an unanchored `models/` gitignore rule. Basic
    # `import verl` is insufficient because it does not traverse this path.
    from verl.models.transformers.monkey_patch import apply_monkey_patch  # noqa: F401
    from verl.models.transformers.qwen2_vl import get_rope_index  # noqa: F401
    from verl.workers.fsdp_workers import ActorRolloutRefWorker  # noqa: F401

    return {
        "cuda": torch.version.cuda,
        "cuda_architectures": architectures,
        "packages": installed,
        "python": sys.version.split()[0],
        "production_imports": [
            "verl.models.transformers.monkey_patch",
            "verl.models.transformers.qwen2_vl",
            "verl.workers.fsdp_workers.ActorRolloutRefWorker",
        ],
        "validated": True,
    }


def main() -> None:
    print(json.dumps(validate_environment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
