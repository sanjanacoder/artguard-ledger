"""Dataset scan: CLIP image embeddings compared against a prebuilt index."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

import config
from core.models import MatchResult


def load_rgb(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


class Embedder:
    """OpenCLIP image encoder. Loads once per process."""

    def __init__(self, model_name: str = config.CLIP_MODEL, pretrained: str = config.CLIP_PRETRAINED):
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        self.name = f"{model_name} ({pretrained})"

    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = self.torch.stack([self.preprocess(im.convert("RGB")) for im in images[i:i + batch_size]])
                feats = self.model.encode_image(batch).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@dataclass
class ClipIndex:
    embeddings: np.ndarray  # (N, D) float32, L2-normalized
    meta: list[dict]  # one entry per row: id, title, artist, source, license, url, phash, thumb
    info: dict  # name, description, sources, built_at, size

    @classmethod
    def load(cls, index_dir: Path = config.INDEX_DIR) -> "ClipIndex":
        emb = np.load(index_dir / "embeddings.npy").astype(np.float32)
        meta = json.loads((index_dir / "meta.json").read_text())
        info = json.loads((index_dir / "info.json").read_text())
        if len(meta) != len(emb):
            raise ValueError("Index is inconsistent: meta.json and embeddings.npy differ in length")
        return cls(emb, meta, info)

    def __len__(self) -> int:
        return len(self.meta)

    def search(self, query: np.ndarray, k: int = config.TOP_K) -> list[tuple[int, float]]:
        sims = self.embeddings @ query.astype(np.float32)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]


def phash_hex(im: Image.Image) -> str:
    return str(imagehash.phash(im))


def classify(top_score: float, phash_distance: int | None,
             t_strong: float = config.T_STRONG, t_possible: float = config.T_POSSIBLE) -> tuple[str, bool]:
    """Return (label, upgraded_by_phash)."""
    if top_score >= t_strong:
        return "strong", False
    if top_score >= t_possible:
        if phash_distance is not None and phash_distance <= config.PHASH_MAX_DISTANCE:
            return "strong", True
        return "possible", False
    return "none", False


def match_image(im: Image.Image, embedding: np.ndarray, index: ClipIndex, k: int = config.TOP_K) -> MatchResult:
    hits = index.search(embedding, k)
    neighbors = []
    for i, score in hits:
        m = index.meta[i]
        neighbors.append({
            "id": m["id"], "score": round(score, 4), "title": m.get("title"),
            "artist": m.get("artist"), "source": m.get("source"), "license": m.get("license"),
            "url": m.get("url"), "thumb": m.get("thumb"),
        })
    top_i, top_score = hits[0]
    distance = None
    top_hash = index.meta[top_i].get("phash")
    if top_hash:
        distance = int(imagehash.phash(im) - imagehash.hex_to_hash(top_hash))
    label, upgraded = classify(top_score, distance)
    return MatchResult(label, round(top_score, 4), neighbors, distance, upgraded)
