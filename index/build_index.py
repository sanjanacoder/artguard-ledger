"""Build the demo search index.

    python index/build_index.py --count 3000 --holdout 300

1. Downloads public-domain artworks (CC0) from the Art Institute of Chicago API.
2. Adds simulated "scraped" copies of three demo pieces (see core/transforms.py).
3. Embeds everything with CLIP and writes embeddings.npy, meta.json, info.json
   and small thumbnails. Held-out images (not indexed) are kept in index/raw/holdout
   for threshold calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from core.scan import get_embedder, load_rgb, phash_hex  # noqa: E402
from core.transforms import SEEDED_COPIES  # noqa: E402

API = "https://api.artic.edu/api/v1/artworks"
IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg"
HEADERS = {"User-Agent": "artguard-ledger (Banana Hacks 2026 student project)",
           "AIC-User-Agent": "artguard-ledger (Banana Hacks 2026 student project)"}
RAW = config.INDEX_DIR / "raw"
THUMBS = config.INDEX_DIR / "thumbs"


def fetch_records(n: int) -> list[dict]:
    """Public-domain artworks with images. The search endpoint caps results at
    1,000, so page through the plain listing and filter here."""
    records, seen = [], set()
    page = 1
    while len(records) < n:
        params = {"fields": "id,title,artist_title,image_id,is_public_domain,artwork_type_title",
                  "limit": 100, "page": page}
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 403:  # pagination limit reached
            break
        r.raise_for_status()
        data = r.json()["data"]
        if not data:
            break
        for d in data:
            if d.get("is_public_domain") and d.get("image_id") and d["id"] not in seen:
                seen.add(d["id"])
                records.append(d)
        page += 1
        time.sleep(0.7)  # be polite to the API
    print(f"found {len(records)} public-domain records in {page - 1} pages")
    return records[:n]


def download(rec: dict, folder: Path) -> Path | None:
    path = folder / f"aic_{rec['id']}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return path
    for attempt in range(3):
        try:
            r = requests.get(IIIF.format(image_id=rec["image_id"]), headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                path.write_bytes(r.content)
                return path
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def download_all(records: list[dict], folder: Path) -> list[tuple[dict, Path]]:
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda r: download(r, folder), records))
    ok = [(r, p) for r, p in zip(records, paths) if p is not None]
    print(f"downloaded {len(ok)}/{len(records)} into {folder}")
    return ok


def make_seeded() -> list[tuple[dict, Image.Image]]:
    out = []
    for name, (how, fn) in SEEDED_COPIES.items():
        im = load_rgb((config.DEMO_DIR / name).read_bytes())
        copy = fn(im)
        stem = Path(name).stem
        meta = {"id": f"seeded_{stem}", "title": f"Scraped copy of '{stem}'", "artist": "Demo artist",
                "source": "Simulated scraped copy (demo)", "license": "Demo image created by the project team",
                "url": None, "note": how}
        out.append((meta, copy))
    return out


def reseed() -> None:
    """Swap in new simulated copies without re-downloading or re-embedding the rest."""
    emb = np.load(config.INDEX_DIR / "embeddings.npy")
    meta = json.loads((config.INDEX_DIR / "meta.json").read_text())
    keep = [i for i, m in enumerate(meta) if not m["id"].startswith("seeded_")]
    seeded = make_seeded()
    new_emb = get_embedder().embed([im for _m, im in seeded]).astype(np.float16)
    new_meta = []
    for m, im in seeded:
        thumb = im.copy()
        thumb.thumbnail((160, 160))
        thumb.save(THUMBS / f"{m['id']}.jpg", quality=72)
        new_meta.append({**m, "phash": phash_hex(im), "thumb": f"thumbs/{m['id']}.jpg"})
    np.save(config.INDEX_DIR / "embeddings.npy", np.concatenate([emb[keep], new_emb]))
    (config.INDEX_DIR / "meta.json").write_text(json.dumps([meta[i] for i in keep] + new_meta, ensure_ascii=False))
    print(f"replaced simulated copies; index has {len(keep) + len(new_meta)} images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--reseed", action="store_true",
                    help="only replace the simulated copies in an existing index")
    args = ap.parse_args()
    if args.reseed:
        reseed()
        return

    print("fetching records…")
    records = fetch_records(args.count + args.holdout)
    index_recs, holdout_recs = records[:args.count], records[args.count:]
    indexed = download_all(index_recs, RAW / "aic")
    download_all(holdout_recs, RAW / "holdout")

    THUMBS.mkdir(parents=True, exist_ok=True)
    items: list[tuple[dict, Image.Image]] = []
    for rec, path in indexed:
        try:
            im = load_rgb(path.read_bytes())
        except Exception as exc:  # corrupt download
            print("skip", path.name, exc)
            continue
        items.append(({
            "id": f"aic_{rec['id']}", "title": rec.get("title"), "artist": rec.get("artist_title"),
            "source": "Art Institute of Chicago (public domain)", "license": "CC0",
            "url": f"https://www.artic.edu/artworks/{rec['id']}",
        }, im))
    items.extend(make_seeded())

    print(f"embedding {len(items)} images…")
    embedder = get_embedder()
    embs = []
    batch = 64
    for i in range(0, len(items), batch):
        embs.append(embedder.embed([im for _m, im in items[i:i + batch]]))
        print(f"  {min(i + batch, len(items))}/{len(items)}", end="\r")
    emb = np.concatenate(embs).astype(np.float16)

    meta = []
    for m, im in items:
        thumb = im.copy()
        thumb.thumbnail((160, 160))
        thumb.save(THUMBS / f"{m['id']}.jpg", quality=72)
        meta.append({**m, "phash": phash_hex(im), "thumb": f"thumbs/{m['id']}.jpg"})

    np.save(config.INDEX_DIR / "embeddings.npy", emb)
    (config.INDEX_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    info = {
        "name": "ArtGuard demo index",
        "size": len(meta),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": embedder.name,
        "sources": [
            {"name": "Art Institute of Chicago Open Access (public domain, CC0)",
             "count": sum(1 for m in meta if m["id"].startswith("aic_")),
             "url": "https://api.artic.edu/docs/"},
            {"name": "Simulated scraped copies of demo pieces", "count": len(SEEDED_COPIES), "url": None},
        ],
        "description": "A small sample index for demonstration. It is not a real AI training dataset.",
    }
    (config.INDEX_DIR / "info.json").write_text(json.dumps(info, indent=2))
    print(f"\nwrote index with {len(meta)} images")


if __name__ == "__main__":
    main()
