#!/usr/bin/env python3
"""Apply explicit reasoning-output instructions to MathVista."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "FVE reasoning prompt: MathVista"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_vqa_file", type=Path)
    args = parser.parse_args()
    image_vqa_file = args.image_vqa_file.expanduser().resolve()
    source = image_vqa_file.read_text()
    if MARKER in source:
        print(f"VLMEvalKit MathVista reasoning prompt already applied: {image_vqa_file}")
        return

    anchor = "    # It returns a DataFrame\n    @classmethod\n    def evaluate(self, eval_file, **judge_kwargs):\n"
    replacement = '''    def build_prompt(self, line):
        # FVE reasoning prompt: MathVista
        msgs = super().build_prompt(line)
        assert msgs[-1]["type"] == "text"
        msgs[-1]["value"] += (
            "\\nThink step by step and preserve the complete reasoning between "
            "<think> and </think>. End with a final line in exactly this format: "
            "<answer>final answer</answer>."
        )
        return msgs

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
'''
    mathvista_start = source.find("class MathVista(ImageBaseDataset):")
    if mathvista_start < 0:
        raise SystemExit(f"Pinned VLMEvalKit MathVista class is missing: {image_vqa_file}")
    anchor_position = source.find(anchor, mathvista_start)
    if anchor_position < 0:
        raise SystemExit(f"Pinned VLMEvalKit MathVista evaluate block changed: {image_vqa_file}")
    image_vqa_file.write_text(source[:anchor_position] + source[anchor_position:].replace(anchor, replacement, 1))
    print(f"Applied MathVista reasoning prompt patch: {image_vqa_file}")


if __name__ == "__main__":
    main()
