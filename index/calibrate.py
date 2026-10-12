"""Measure CLIP similarities to choose match thresholds and build the robustness chart.

    python index/calibrate.py            # print stats, write robustness.json
    python index/calibrate.py --write    # also write calibration.json (thresholds)

Positives: an image vs. its own edited copies (crop, resize, recompress...).
Negatives: held-out images (never indexed) vs. their nearest neighbor in the index.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from core.scan import ClipIndex, get_embedder, load_rgb  # noqa: E402
from core.transforms import ROBUSTNESS_VARIANTS  # noqa: E402

HOLDOUT = config.INDEX_DIR / "raw" / "holdout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write thresholds to calibration.json")
    ap.add_argument("--positives", type=int, default=40, help="held-out images used as positives")
    args = ap.parse_args()

    embedder = get_embedder()
    index = ClipIndex.load()
    unseeded = np.array([not m["id"].startswith("seeded_") for m in index.meta])

    holdout_files = sorted(HOLDOUT.glob("*.jpg"))
    holdout = [load_rgb(f.read_bytes()) for f in holdout_files]
    h_emb = embedder.embed(holdout)
    negatives = (h_emb @ index.embeddings[unseeded].T).max(1)

    random.seed(0)
    demo = [load_rgb(f.read_bytes()) for f in sorted(config.DEMO_DIR.iterdir()) if f.suffix in (".jpg", ".png", ".webp")]
    sample = demo + random.sample(holdout, min(args.positives, len(holdout)))
    base = embedder.embed(sample)
    per_variant: dict[str, list[float]] = {}
    for label, fn in ROBUSTNESS_VARIANTS[1:]:
        emb = embedder.embed([fn(im) for im in sample])
        per_variant[label] = (emb * base).sum(1).tolist()
    positives = np.array([v for vals in per_variant.values() for v in vals])

    def pct(a, q):
        return round(float(np.percentile(a, q)), 4)

    print(f"negatives (n={len(negatives)}): p50 {pct(negatives, 50)}  p95 {pct(negatives, 95)}  "
          f"p99 {pct(negatives, 99)}  max {pct(negatives, 100)}")
    for label, vals in per_variant.items():
        print(f"  {label:34s} min {min(vals):.3f}  p10 {np.percentile(vals, 10):.3f}  mean {np.mean(vals):.3f}")

    t_possible = round(max(pct(negatives, 99), 0.75), 2)
    t_strong = round(max(pct(negatives, 99.9) + 0.02, t_possible + 0.06), 2)
    caught_strong = float((positives >= t_strong).mean())
    caught_any = float((positives >= t_possible).mean())
    false_any = float((negatives >= t_possible).mean())
    print(f"suggested: strong >= {t_strong}, possible >= {t_possible}")
    print(f"positives caught: strong {caught_strong:.0%}, strong+possible {caught_any:.0%}; "
          f"negatives flagged: {false_any:.1%}")

    # Robustness chart: one demo piece vs. its edits, plus the nearest unrelated image.
    piece_file = config.DEMO_DIR / "tide_circles.jpg"
    piece = load_rgb(piece_file.read_bytes())
    piece_emb = embedder.embed([piece])[0]
    rows = []
    for label, fn in ROBUSTNESS_VARIANTS:
        emb = embedder.embed([fn(piece)])[0]
        rows.append({"variant": label, "similarity": round(float(emb @ piece_emb), 4)})
    nearest_unrelated = float((index.embeddings[unseeded] @ piece_emb).max())
    rows.append({"variant": "Nearest unrelated image in index", "similarity": round(nearest_unrelated, 4)})
    robustness = {
        "piece": piece_file.name,
        "rows": rows,
        "t_strong": t_strong if args.write else config.T_STRONG,
        "t_possible": t_possible if args.write else config.T_POSSIBLE,
        "variant_summary": {k: {"min": round(min(v), 4), "mean": round(float(np.mean(v)), 4)}
                            for k, v in per_variant.items()},
        "n_positive_images": len(sample),
    }
    (config.INDEX_DIR / "robustness.json").write_text(json.dumps(robustness, indent=1))

    if args.write:
        (config.INDEX_DIR / "calibration.json").write_text(json.dumps({
            "t_strong": t_strong, "t_possible": t_possible,
            "negatives": {"n": len(negatives), "p50": pct(negatives, 50), "p95": pct(negatives, 95),
                          "p99": pct(negatives, 99), "max": pct(negatives, 100)},
            "positives": {"n": len(positives), "caught_strong": caught_strong, "caught_any": caught_any},
            "false_positive_rate": false_any,
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=1))
        print("wrote calibration.json")


if __name__ == "__main__":
    main()
