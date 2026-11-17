#!/usr/bin/env python3
"""Add official DeepSeek judge routing and redact API-key logging."""

from __future__ import annotations

import argparse
from pathlib import Path


JUDGE_MARKER = "'deepseek-chat': 'deepseek-chat'"
REDACTION_MARKER = "API Key: ***REDACTED***"


def patch_judge_util(path: Path) -> None:
    source = path.read_text()
    if JUDGE_MARKER not in source:
        anchor = "            'deepseek': 'deepseek-ai/DeepSeek-V2.5',\n"
        if source.count(anchor) != 1:
            raise SystemExit(f"Pinned VLMEvalKit judge map changed: {path}")
        source = source.replace(anchor, anchor + "            'deepseek-chat': 'deepseek-chat',\n", 1)
        path.write_text(source)


def patch_gpt_wrapper(path: Path) -> None:
    source = path.read_text()
    if REDACTION_MARKER not in source:
        anchor = "        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')\n"
        replacement = "        self.logger.info(f'Using API Base: {self.api_base}; API Key: ***REDACTED***')\n"
        if source.count(anchor) != 1:
            raise SystemExit(f"Pinned VLMEvalKit API logging block changed: {path}")
        source = source.replace(anchor, replacement, 1)
        path.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judge_util", type=Path)
    parser.add_argument("gpt_wrapper", type=Path)
    args = parser.parse_args()
    judge_util = args.judge_util.expanduser().resolve()
    gpt_wrapper = args.gpt_wrapper.expanduser().resolve()
    patch_judge_util(judge_util)
    patch_gpt_wrapper(gpt_wrapper)
    print(f"Applied official DeepSeek judge patch: {judge_util}")
    print(f"Redacted VLMEvalKit API-key logging: {gpt_wrapper}")


if __name__ == "__main__":
    main()
