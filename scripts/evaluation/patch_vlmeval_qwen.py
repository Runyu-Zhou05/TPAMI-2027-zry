#!/usr/bin/env python3
"""Apply the scoped Qwen config-detection fix to pinned VLMEvalKit."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "FVE patch: exported checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=Path)
    args = parser.parse_args()
    model_file = args.model_file.expanduser().resolve()
    source = model_file.read_text()
    if MARKER in source:
        print(f"VLMEvalKit Qwen patch already applied: {model_file}")
        return

    import_anchor = "import os\n"
    if source.count(import_anchor) != 1:
        raise SystemExit(f"Unexpected import layout in {model_file}")
    source = source.replace(import_anchor, "import json\nimport os\n", 1)

    old = """        if '2.5' in model_path:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
"""
    new = """        # FVE patch: exported checkpoints need not contain \"2.5\" in their
        # directory name. Detect the architecture from config.json instead.
        with open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8') as handle:
            model_config = json.load(handle)
        architectures = ' '.join(model_config.get('architectures') or []).lower()
        model_type = str(model_config.get('model_type', '')).lower()
        if 'qwen2_5' in architectures or model_type == 'qwen2_5_vl':
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
"""
    if source.count(old) != 1:
        raise SystemExit(f"Pinned VLMEvalKit Qwen detection block changed in {model_file}")
    model_file.write_text(source.replace(old, new, 1))
    print(f"Applied VLMEvalKit Qwen config-detection patch: {model_file}")


if __name__ == "__main__":
    main()
