"""Runs the workflow on a batch and records every step in the ActionLog."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError

import config
from core import rights, scoring, tagging
from core.hashing import sha256_bytes
from core.models import ActionLog, Piece, utc_now
from core.scan import ClipIndex, Embedder, load_rgb, match_image

MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


@dataclass
class Batch:
    artist: str = ""
    website: str = ""
    pieces: list[Piece] = field(default_factory=list)
    log: ActionLog = field(default_factory=ActionLog)
    created_at: str = field(default_factory=utc_now)
    index_info: dict | None = None
    model_name: str | None = None


class RejectedFile(ValueError):
    pass


def ingest(batch: Batch, filename: str, data: bytes) -> Piece:
    """Validate an upload, hash it, and add it to the batch."""
    if len(batch.pieces) >= config.MAX_FILES:
        raise RejectedFile(f"Batch limit is {config.MAX_FILES} files.")
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise RejectedFile(f"{filename} is larger than {config.MAX_FILE_MB} MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise RejectedFile(f"{filename} is not a readable image ({exc}).") from exc
    if fmt not in MIME:
        raise RejectedFile(f"{filename} is {fmt}; only JPEG, PNG and WebP are supported.")
    piece = Piece(filename=_unique_name(batch, filename), mime=MIME[fmt], original_bytes=data,
                  sha256_original=sha256_bytes(data), width=w, height=h)
    batch.pieces.append(piece)
    batch.log.add("hash", "done_by_app", f"Received {piece.filename} ({w}x{h}, {len(data):,} bytes). "
                  f"SHA-256 {piece.sha256_original}", piece.id)
    return piece


def _unique_name(batch: Batch, filename: str) -> str:
    taken = {p.filename for p in batch.pieces}
    if filename not in taken:
        return filename
    stem, dot, ext = filename.rpartition(".")
    n = 2
    while f"{stem}-{n}{dot}{ext}" in taken:
        n += 1
    return f"{stem}-{n}{dot}{ext}"


def scan(batch: Batch, index: ClipIndex, embedder: Embedder) -> None:
    batch.index_info = index.info
    batch.model_name = embedder.name
    images = [load_rgb(p.original_bytes) for p in batch.pieces]
    embeddings = embedder.embed(images)
    for piece, im, emb in zip(batch.pieces, images, embeddings):
        piece.match = match_image(im, emb, index)
        m = piece.match
        detail = (f"Searched {len(index):,} images in '{index.info['name']}'. Result: {m.label} match "
                  f"(top similarity {m.top_score:.3f}, nearest: {m.neighbors[0]['title']!r}).")
        if m.upgraded_by_phash:
            detail += f" Upgraded to strong by perceptual hash (distance {m.phash_distance})."
        batch.log.add("scan", "done_by_app", detail, piece.id)


def tag(batch: Batch, engine: str = "auto") -> None:
    for piece in batch.pieces:
        try:
            result = tagging.add_dnt_tag(piece.original_bytes, batch.artist or None, batch.website or None, engine)
        except Exception as exc:  # keep going; record the failure
            batch.log.add("tag", "failed", f"Could not write the Do Not Train tag: {exc}", piece.id)
            continue
        piece.tagged_bytes = result.data
        piece.sha256_tagged = sha256_bytes(result.data)
        piece.tag_engine = result.engine
        piece.tag_verified = result.verified
        batch.log.add("tag", "done_by_app",
                      f"Wrote XMP plus:DataMining = {config.DMI_PROHIBITED_AIML} to a copy "
                      f"(engine: {result.engine}). Tagged copy SHA-256 {piece.sha256_tagged}", piece.id)
        batch.log.add("verify_tag", "done_by_app" if result.verified else "failed",
                      f"Read the tag back from the copy: {result.data_mining_value or 'not found'}", piece.id)


def assess_rights(batch: Batch) -> None:
    for piece in batch.pieces:
        summary = rights.summarize(piece.ai_answers)
        piece.rights_key = summary["key"]
        piece.rights_summary = summary["summary"]
        batch.log.add("rights", "done_by_app", f"Rights summary: {summary['status']} ({summary['key']}).", piece.id)


def score(batch: Batch) -> None:
    for piece in batch.pieces:
        scoring.apply(piece)
        parts = ", ".join(f"{k} +{v}" for k, v in piece.breakdown.items())
        batch.log.add("score", "done_by_app", f"Exposure score {piece.score} ({piece.label}): {parts}.", piece.id)


def verify_originals(batch: Batch) -> bool:
    """Re-hash every original to show it was never modified."""
    ok = True
    for piece in batch.pieces:
        same = sha256_bytes(piece.original_bytes) == piece.sha256_original
        ok &= same
        batch.log.add("verify_original", "done_by_app" if same else "failed",
                      "Original re-hashed: unchanged." if same else "Original hash changed!", piece.id)
    return ok


def run(batch: Batch, index: ClipIndex, embedder: Embedder, tag_engine: str = "auto") -> None:
    batch.log.add("start", "done_by_app", f"{config.APP_NAME} {config.APP_VERSION}: processing "
                  f"{len(batch.pieces)} piece(s) for {batch.artist or 'an unnamed artist'}.")
    scan(batch, index, embedder)
    tag(batch, tag_engine)
    assess_rights(batch)
    score(batch)
    verify_originals(batch)


def thumbnail_jpeg(data: bytes, size: int = 256) -> bytes:
    im = load_rgb(data)
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return buf.getvalue()
