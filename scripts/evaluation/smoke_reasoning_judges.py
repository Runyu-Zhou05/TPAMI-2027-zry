#!/usr/bin/env python3
"""Smoke MathVista DeepSeek extraction and MMMU exact matching without a VLM."""

from __future__ import annotations

import argparse
import json
import re

import requests

from deepseek_credential import resolve_deepseek_api_key


def deepseek_extract(api_base: str, api_key: str, timeout: int) -> str:
    synthetic_response = "<think>Three rows with four objects each gives 3 * 4 = 12.</think> <answer>12</answer>"
    prompt = (
        "Extract the final answer from the candidate response to this MathVista-style question. "
        "Return only the answer, with no explanation.\n"
        "Question: How many objects are shown if there are three rows of four objects?\n"
        f"Candidate response: {synthetic_response}"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16,
    }
    try:
        response = requests.post(
            api_base,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if not response.ok:
            raise SystemExit(f"DeepSeek judge smoke failed with HTTP {response.status_code}")
        result = response.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        raise SystemExit(f"DeepSeek judge smoke request failed: {type(exc).__name__}") from exc
    try:
        extracted = str(result["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit("DeepSeek judge smoke returned an unexpected response schema") from exc
    if extracted.strip("` \n\t.") != "12":
        raise SystemExit(f"DeepSeek judge extracted an unexpected MathVista answer: {extracted!r}")
    return "12"


def mmmu_exact_match() -> str:
    synthetic_response = "<think>The second option is consistent with the diagram.</think> <answer>B</answer>"
    matches = re.findall(r"<answer>\s*([A-D])\s*</answer>", synthetic_response)
    if matches != ["B"]:
        raise SystemExit("MMMU smoke could not extract exactly one answer tag")
    if matches[0] != "B":
        raise SystemExit("MMMU exact-match smoke failed")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-api-base", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    api_key, credential_source = resolve_deepseek_api_key()
    result = {
        "credential_source": credential_source,
        "mathvista": {"judge": "deepseek-chat", "extracted_answer": deepseek_extract(args.deepseek_api_base, api_key, args.timeout)},
        "mmmu": {"judge": "exact_matching", "extracted_answer": mmmu_exact_match()},
        "model_loaded": False,
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
