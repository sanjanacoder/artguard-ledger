"""Download public-domain reference works for the audit's demo artist.

    python audit/fetch_reference.py --max 80

Uses the Art Institute of Chicago API (CC0 images). Writes audit/reference/*.jpg
and audit/reference/sources.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

SEARCH = "https://api.artic.edu/api/v1/artworks/search"
IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg"
HEADERS = {"AIC-User-Agent": "artguard-ledger (Banana Hacks 2026 student project)"}
REF = config.AUDIT_DIR / "reference"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=80)
    args = ap.parse_args()
    setup = yaml.safe_load((config.AUDIT_DIR / "prompts.yaml").read_text())
    REF.mkdir(parents=True, exist_ok=True)
    found = []
    for page in range(1, 10):
        r = requests.get(SEARCH, params={"q": setup["reference_query"], "limit": 100, "page": page,
                                         "fields": "id,title,artist_title,image_id,is_public_domain"},
                         headers=HEADERS, timeout=30)
        if r.status_code != 200 or not r.json()["data"]:
            break
        for d in r.json()["data"]:
            if (d.get("is_public_domain") and d.get("image_id")
                    and setup["reference_query"].lower() in (d.get("artist_title") or "").lower()):
                found.append(d)
        time.sleep(0.7)
    found = found[:args.max]
    sources = []
    for d in found:
        path = REF / f"aic_{d['id']}.jpg"
        if not path.exists():
            img = requests.get(IIIF.format(image_id=d["image_id"]), headers=HEADERS, timeout=30)
            if img.status_code != 200:
                continue
            path.write_bytes(img.content)
            time.sleep(0.2)
        sources.append({"file": path.name, "title": d["title"], "artist": d["artist_title"],
                        "url": f"https://www.artic.edu/artworks/{d['id']}", "license": "CC0"})
    (REF / "sources.json").write_text(json.dumps(sources, indent=1, ensure_ascii=False))
    print(f"saved {len(sources)} reference images to {REF}")


if __name__ == "__main__":
    main()
