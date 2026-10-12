"""Image edits that mimic how scraped copies differ from the original."""
import io

from PIL import Image, ImageEnhance, ImageOps


def jpeg(im: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def crop(im: Image.Image, fraction: float, anchor: str = "center") -> Image.Image:
    """Remove `fraction` of each dimension (e.g. 0.25 keeps 75% x 75%)."""
    w, h = im.size
    cw, ch = int(w * (1 - fraction)), int(h * (1 - fraction))
    if anchor == "topleft":
        x, y = 0, 0
    else:
        x, y = (w - cw) // 2, (h - ch) // 2
    return im.crop((x, y, x + cw, y + ch))


def resize(im: Image.Image, scale: float) -> Image.Image:
    return im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)


def hue_shift(im: Image.Image, shift: int) -> Image.Image:
    h, s, v = im.convert("HSV").split()
    h = h.point(lambda p: (p + shift) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def brightness(im: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(factor)


def grayscale(im: Image.Image) -> Image.Image:
    return ImageOps.grayscale(im).convert("RGB")


def border(im: Image.Image, fraction: float = 0.08, color=(255, 255, 255)) -> Image.Image:
    """Screenshot-style padding around the image."""
    pad = int(min(im.size) * fraction)
    return ImageOps.expand(im, border=pad, fill=color)


def flip(im: Image.Image) -> Image.Image:
    return ImageOps.mirror(im)


# Named variants used for the robustness chart (label, function).
ROBUSTNESS_VARIANTS = [
    ("Original", lambda im: im),
    ("Resize 50%", lambda im: resize(im, 0.5)),
    ("Resize 25%", lambda im: resize(im, 0.25)),
    ("JPEG q90", lambda im: jpeg(im, 90)),
    ("JPEG q60", lambda im: jpeg(im, 60)),
    ("JPEG q30", lambda im: jpeg(im, 30)),
    ("Crop 10%", lambda im: crop(im, 0.10)),
    ("Crop 25%", lambda im: crop(im, 0.25)),
    ("Crop 40%", lambda im: crop(im, 0.40)),
    ("Hue shift", lambda im: hue_shift(im, 40)),
    ("Grayscale", grayscale),
    ("Mirror", flip),
    ("Screenshot border", border),
    ("Crop 25% + resize 50% + JPEG q60", lambda im: jpeg(resize(crop(im, 0.25), 0.5), 60)),
]

# How the demo pieces were "scraped" into the index (see index/build_index.py).
SEEDED_COPIES = {
    "tide_circles.jpg": ("Resized 50%, JPEG q75",
                         lambda im: jpeg(resize(im, 0.5), 75)),
    "paper_mountains.jpg": ("Cropped 15%, brightened, JPEG q60",
                            lambda im: jpeg(brightness(crop(im, 0.15, "topleft"), 1.15), 60)),
    "mosaic_sun.webp": ("Cropped 30%, hue shifted, JPEG q50",
                        lambda im: jpeg(hue_shift(crop(im, 0.30, "topleft"), 25), 50)),
}
