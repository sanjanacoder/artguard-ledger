import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_image(fmt: str, mode: str = "RGB", size=(96, 64)) -> bytes:
    color = (30, 120, 200) if mode == "RGB" else (30, 120, 200, 128)
    im = Image.new(mode, size, color)
    for x in range(0, size[0], 8):  # some structure, so codecs have real work to do
        for y in range(size[1]):
            im.putpixel((x, y), (250, 200, 20) if mode == "RGB" else (250, 200, 20, 255))
    buf = io.BytesIO()
    im.save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes():
    return make_image("JPEG")


@pytest.fixture(params=[("JPEG", "RGB"), ("PNG", "RGB"), ("PNG", "RGBA"), ("WEBP", "RGB"), ("WEBP", "RGBA")],
                ids=lambda p: f"{p[0]}-{p[1]}")
def image_bytes(request):
    fmt, mode = request.param
    return make_image(fmt, mode)
