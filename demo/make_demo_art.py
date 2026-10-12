"""Generate the demo artist's portfolio procedurally, so we own every pixel.

    python demo/make_demo_art.py

Writes five 1024x1024 pieces to demo/artist_samples/. Deterministic.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SIZE = 1024


def tide_circles(rng: random.Random) -> Image.Image:
    im = Image.new("RGB", (SIZE, SIZE), (12, 34, 56))
    layer = Image.new("RGBA", (SIZE, SIZE))
    d = ImageDraw.Draw(layer)
    palette = [(32, 140, 160), (90, 200, 190), (240, 230, 200), (20, 90, 130), (250, 160, 90)]
    for _ in range(140):
        r = rng.randint(30, 220)
        x, y = rng.randint(-50, SIZE + 50), rng.randint(-50, SIZE + 50)
        c = rng.choice(palette)
        d.ellipse([x - r, y - r, x + r, y + r], fill=c + (rng.randint(40, 110),), outline=c + (200,), width=3)
    im.paste(layer, (0, 0), layer)
    return im


def flow_field(rng: random.Random) -> Image.Image:
    im = Image.new("RGB", (SIZE, SIZE), (250, 244, 232))
    d = ImageDraw.Draw(im)
    seed = rng.random() * 10
    warm = [(200, 60, 40), (230, 140, 50), (120, 40, 60), (40, 40, 40)]
    for _ in range(900):
        x, y = rng.uniform(0, SIZE), rng.uniform(0, SIZE)
        c = rng.choice(warm)
        pts = [(x, y)]
        for _ in range(60):
            a = math.sin(x * 0.006 + seed) * 2.2 + math.cos(y * 0.005 - seed) * 2.0
            x, y = x + math.cos(a) * 6, y + math.sin(a) * 6
            pts.append((x, y))
        d.line(pts, fill=c, width=2)
    return im


def bauhaus_blocks(rng: random.Random) -> Image.Image:
    im = Image.new("RGB", (SIZE, SIZE), (238, 232, 218))
    d = ImageDraw.Draw(im)
    colors = [(214, 40, 40), (247, 191, 30), (30, 70, 160), (20, 20, 20), (238, 232, 218)]
    cell = SIZE // 8
    for gx in range(8):
        for gy in range(8):
            x0, y0 = gx * cell, gy * cell
            kind = rng.randint(0, 3)
            c = rng.choice(colors)
            if kind == 0:
                d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=c)
            elif kind == 1:
                d.ellipse([x0 + 8, y0 + 8, x0 + cell - 8, y0 + cell - 8], fill=c)
            elif kind == 2:
                d.pieslice([x0, y0, x0 + 2 * cell, y0 + 2 * cell], 180, 270, fill=c)
            else:
                d.polygon([(x0, y0 + cell), (x0 + cell, y0 + cell), (x0 + cell, y0)], fill=c)
    for i in range(9):
        d.line([(i * cell, 0), (i * cell, SIZE)], fill=(20, 20, 20), width=6)
        d.line([(0, i * cell), (SIZE, i * cell)], fill=(20, 20, 20), width=6)
    return im


def paper_mountains(rng: random.Random) -> Image.Image:
    y = np.linspace(0, 1, SIZE)[:, None]
    sky = np.concatenate([
        (250 - 60 * y) * np.ones((1, SIZE)),
        (190 - 40 * y) * np.ones((1, SIZE)),
        (150 + 60 * y) * np.ones((1, SIZE)),
    ]).reshape(3, SIZE, SIZE).transpose(1, 2, 0)
    im = Image.fromarray(sky.clip(0, 255).astype("uint8"))
    d = ImageDraw.Draw(im)
    d.ellipse([640, 140, 820, 320], fill=(255, 240, 210))
    for layer in range(6):
        base = 420 + layer * 95
        shade = (120 - layer * 18, 70 + layer * 6, 110 - layer * 10)
        pts = [(0, SIZE)]
        phase = rng.random() * 6
        for x in range(0, SIZE + 16, 16):
            h = base - 120 * abs(math.sin(x * 0.004 + phase)) - 40 * math.sin(x * 0.02 + layer)
            pts.append((x, h))
        pts.append((SIZE, SIZE))
        d.polygon(pts, fill=shade)
    return im.filter(ImageFilter.SMOOTH)


def mosaic_sun(rng: random.Random) -> Image.Image:
    n = 260
    pts = np.array([[rng.uniform(0, SIZE), rng.uniform(0, SIZE)] for _ in range(n)])
    yy, xx = np.mgrid[0:SIZE:2, 0:SIZE:2]
    grid = np.stack([xx.ravel(), yy.ravel()], 1)
    nearest = np.empty(len(grid), dtype=int)
    for i in range(0, len(grid), 65536):
        chunk = grid[i:i + 65536]
        nearest[i:i + 65536] = ((chunk[:, None, :] - pts[None]) ** 2).sum(-1).argmin(1)
    center = np.array([SIZE / 2, SIZE / 2])
    dist = np.linalg.norm(pts - center, axis=1) / (SIZE * 0.7)
    cols = np.stack([255 - 80 * dist, 200 - 170 * dist, 60 + 120 * dist], 1)
    cols += np.array([[rng.uniform(-25, 25) for _ in range(3)] for _ in range(n)])
    img = cols[nearest].reshape(SIZE // 2, SIZE // 2, 3).clip(0, 255).astype("uint8")
    im = Image.fromarray(img).resize((SIZE, SIZE), Image.NEAREST)
    edges = im.filter(ImageFilter.FIND_EDGES).convert("L").point(lambda v: 255 if v > 10 else 0)
    im.paste((30, 20, 20), (0, 0), edges)
    return im


PIECES = {
    "tide_circles.jpg": tide_circles,
    "flow_field.png": flow_field,
    "bauhaus_blocks.jpg": bauhaus_blocks,
    "paper_mountains.jpg": paper_mountains,
    "mosaic_sun.webp": mosaic_sun,
}


def main() -> None:
    config.DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for i, (name, fn) in enumerate(PIECES.items()):
        im = fn(random.Random(1000 + i))
        path = config.DEMO_DIR / name
        if name.endswith(".jpg"):
            im.save(path, quality=92)
        elif name.endswith(".webp"):
            im.save(path, quality=90)
        else:
            im.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
