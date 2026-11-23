"""Strict labeled VQA reward for Golden RLVR."""

from __future__ import annotations

import re


ANSWER_PATTERN = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
FINAL_ANSWER_PATTERN = re.compile(r"<answer>\s*([A-D])\s*</answer>\s*$", re.IGNORECASE)


def compute_score(predict_str: str, ground_truth: str) -> float:
    matches = ANSWER_PATTERN.findall(str(predict_str))
    if len(matches) != 1:
        return 0.0
    final = FINAL_ANSWER_PATTERN.search(str(predict_str))
    if final is None:
        return 0.0
    answer = str(ground_truth).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid Golden VQA ground truth: {ground_truth!r}")
    return float(final.group(1).upper() == answer)
