#!/usr/bin/env python3
"""Apply the SSL4RL README MCQ response-format prompt to VLMEvalKit."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "The last line of your response should be of the following format:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_mcq_file", type=Path)
    args = parser.parse_args()
    image_mcq_file = args.image_mcq_file.expanduser().resolve()
    source = image_mcq_file.read_text()
    if MARKER in source:
        print(f"VLMEvalKit SSL4RL MCQ prompt already applied: {image_mcq_file}")
        return

    old = "            prompt += 'Please select the correct answer from the options above. \\n'\n"
    new = """            prompt += (
                'Answer this multiple-choice question. Think step by step before answering. '
                'The last line of your response should be of the following format: '
                '<think>step-by-step reasoning</think> <answer>$LETTER</answer>, '
                'where LETTER is one of the options.\\n'
            )
"""
    if source.count(old) != 1:
        raise SystemExit(f"Pinned VLMEvalKit MCQ prompt block changed in {image_mcq_file}")
    image_mcq_file.write_text(source.replace(old, new, 1))
    print(f"Applied SSL4RL MCQ prompt patch: {image_mcq_file}")


if __name__ == "__main__":
    main()
