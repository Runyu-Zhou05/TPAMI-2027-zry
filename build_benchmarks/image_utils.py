"""Image decoding helpers shared by benchmark builders."""

import base64
import binascii
import os
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError


def _load_image_bytes(data: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("The decoded value is not a supported image") from exc


def decode_image(value: Any) -> Image.Image:
    """Return a detached RGB PIL image from common Hugging Face representations.

    Current MMBench parquet files store JPEGs as base64 strings. Older dataset
    revisions and datasets with an ``Image`` feature may instead return a PIL
    image, a path, or a ``{"bytes": ..., "path": ...}`` dictionary.
    """

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            value = value["path"]
        else:
            raise ValueError("Image dictionaries must contain 'bytes' or 'path'")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _load_image_bytes(bytes(value))

    if isinstance(value, os.PathLike):
        value = os.fspath(value)

    if isinstance(value, str):
        # Check short strings as paths first. Calling isfile on a large base64
        # value can itself fail with ENAMETOOLONG.
        if len(value) < 4096:
            try:
                if os.path.isfile(value):
                    with Image.open(value) as image:
                        image.load()
                        return image.convert("RGB")
            except OSError:
                pass

        encoded = value
        if value.startswith("data:"):
            try:
                header, encoded = value.split(",", 1)
            except ValueError as exc:
                raise ValueError("Malformed image data URI") from exc
            if ";base64" not in header:
                raise ValueError("Only base64 image data URIs are supported")

        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Image string is neither an existing path nor valid base64") from exc
        return _load_image_bytes(data)

    raise TypeError(f"Unsupported image representation: {type(value).__name__}")
