import base64
from io import BytesIO

from PIL import Image

from build_benchmarks.image_utils import decode_image


def _encoded_test_image() -> tuple[str, bytes]:
    image = Image.new("RGBA", (7, 5), color=(10, 20, 30, 128))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return base64.b64encode(data).decode("ascii"), data


def test_decode_base64_image():
    encoded, _ = _encoded_test_image()
    image = decode_image(encoded)
    assert image.mode == "RGB"
    assert image.size == (7, 5)


def test_decode_huggingface_bytes_dict():
    _, data = _encoded_test_image()
    image = decode_image({"bytes": data, "path": None})
    assert image.mode == "RGB"
    assert image.size == (7, 5)


def test_decode_pil_image_returns_detached_rgb_copy():
    source = Image.new("L", (3, 2), color=42)
    image = decode_image(source)
    assert image.mode == "RGB"
    assert image.size == (3, 2)
