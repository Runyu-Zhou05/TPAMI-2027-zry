"""Resolve the DeepSeek credential without printing or persisting plaintext."""

from __future__ import annotations

import base64
import os
from pathlib import Path


def _validate(value: str) -> str:
    value = value.strip()
    if not value.startswith("sk-") or any(character.isspace() for character in value):
        raise SystemExit("DeepSeek credential is missing or malformed")
    return value


def resolve_deepseek_api_key(key_file: Path | None = None) -> tuple[str, str]:
    """Return the key and a non-secret provenance label."""
    if key_file is not None:
        path = key_file.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"DeepSeek API-key file is missing: {path}")
        return _validate(path.read_text()), "explicit_key_file"

    environment_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if environment_key:
        return _validate(environment_key), "environment"

    try:
        from deepseek_api_key_b64 import DEEPSEEK_API_KEY_B64

        decoded = base64.b64decode(DEEPSEEK_API_KEY_B64, validate=True).decode("utf-8")
    except (ImportError, ValueError, UnicodeDecodeError) as exc:
        raise SystemExit("Bundled DeepSeek credential is unavailable or malformed") from exc
    return _validate(decoded), "bundled_base64"
