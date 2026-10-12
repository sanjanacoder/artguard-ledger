"""ZIP of tagged copies plus a manifest. Originals are not included or changed."""
from __future__ import annotations

import io
import json
import zipfile

import config
from core.pipeline import Batch


def build_zip(batch: Batch) -> bytes:
    manifest = {
        "app": f"{config.APP_NAME} {config.APP_VERSION}",
        "created_at_utc": batch.created_at,
        "artist": batch.artist or None,
        "website": batch.website or None,
        "tag": {"property": "XMP plus:DataMining", "value": config.DMI_PROHIBITED_AIML, "notice": config.DNT_NOTICE},
        "files": [],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in batch.pieces:
            entry = {"file": p.filename, "sha256_original": p.sha256_original, "sha256_tagged": p.sha256_tagged,
                     "tag_engine": p.tag_engine, "tag_verified": p.tag_verified}
            if p.tagged_bytes is not None:
                zf.writestr(f"tagged/{p.filename}", p.tagged_bytes)
                entry["path"] = f"tagged/{p.filename}"
            manifest["files"].append(entry)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr("README.txt", _readme(batch))
    return buf.getvalue()


def _readme(batch: Batch) -> str:
    limits = "\n".join(f"- {line}" for line in config.LIMITATIONS)
    return f"""{config.APP_NAME}: tagged copies
Created {batch.created_at} (UTC)

The files in tagged/ are copies of your uploads with "Do Not Train" metadata added:
  XMP plus:DataMining = {config.DMI_PROHIBITED_AIML}
  plus:OtherConstraints, xmpRights:UsageTerms and dc:rights = "{config.DNT_NOTICE}"
Pixels are unchanged; only metadata was added. Your original files were not modified.

Check a tag yourself with ExifTool:
  exiftool -XMP-plus:DataMining tagged/<file>

manifest.json lists the SHA-256 hash of each original and each tagged copy.

Limits:
{limits}
"""
