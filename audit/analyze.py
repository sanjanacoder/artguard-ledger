"""Compute the style lift from generated images and write audit/results.json.

    python audit/analyze.py

Style lift = mean CLIP similarity of "in the style of <artist>" outputs to the
artist's reference centroid, minus the same for a control prompt. 95% CIs come
from a paired bootstrap over subjects.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from audit.generate import OUT, load_setup, slug  # noqa: E402

THUMBS = config.AUDIT_DIR / "thumbs"
REF = config.AUDIT_DIR / "reference"
N_BOOT = 1000
GRID_SUBJECT = 0  # which subject to show in the sample grid


def bootstrap_lift(styled: np.ndarray, control: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap over subjects. Inputs: per-subject mean similarity."""
    diff = styled - control
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def is_blank(im: Image.Image) -> bool:
    """Safety-checker output is a black image; leave those out."""
    return float(np.asarray(im.convert("L")).mean()) < 3


def per_subject_means(sims: dict[tuple[str, int, int], float], template: str, n_subjects: int) -> np.ndarray:
    out = []
    for s in range(n_subjects):
        vals = [v for (t, subj, _seed), v in sims.items() if t == template and subj == s]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def analyze(embed) -> dict:
    """`embed` maps a list of PIL images to L2-normalized embeddings."""
    setup = load_setup()
    ref_files = sorted(REF.glob("*.jpg"))
    if not ref_files:
        raise SystemExit("No reference images. Run audit/fetch_reference.py first.")
    ref_emb = embed([Image.open(f).convert("RGB") for f in ref_files])
    centroid = ref_emb.mean(0)
    centroid /= np.linalg.norm(centroid)
    n_subjects = len(setup["subjects"])
    THUMBS.mkdir(parents=True, exist_ok=True)

    models_out, grid_rows = [], []
    for model in setup["models"]:
        folder = OUT / slug(model["id"])
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            print("no outputs for", model["id"])
            continue
        manifest = json.loads(manifest_path.read_text())
        keys, images, skipped = [], [], 0
        for name, m in manifest.items():
            im = Image.open(folder / name).convert("RGB")
            if m.get("flagged") or is_blank(im):
                skipped += 1
                continue
            keys.append((m["template"], m["subject"], m["seed"]))
            images.append(im)
        emb = embed(images)
        to_centroid = emb @ centroid
        to_refs = emb @ ref_emb.T
        sims = dict(zip(keys, to_centroid.tolist()))
        styled = per_subject_means(sims, "styled", n_subjects)
        entry = {"model": model["label"], "model_id": model["id"], "steps": model["steps"],
                 "guidance": model["guidance"], "skipped_images": skipped,
                 "n_styled": sum(1 for k in keys if k[0] == "styled"),
                 "mean_similarity": {t: float(np.nanmean(per_subject_means(sims, t, n_subjects)))
                                     for t in setup["templates"]}}
        for control, prefix in (("generic", ""), ("genre", "genre_")):
            ok = ~np.isnan(styled) & ~np.isnan(per_subject_means(sims, control, n_subjects))
            lift, lo, hi = bootstrap_lift(styled[ok], per_subject_means(sims, control, n_subjects)[ok])
            entry.update({f"{prefix}lift": lift, f"{prefix}ci_low": lo, f"{prefix}ci_high": hi})
        styled_mask = np.array([k[0] == "styled" for k in keys])
        nearest = to_refs[styled_mask].max(1) if styled_mask.any() else np.array([])
        entry["near_copy_rate"] = float((nearest >= config.T_POSSIBLE).mean()) if len(nearest) else 0.0
        entry["max_similarity_to_a_reference"] = float(nearest.max()) if len(nearest) else 0.0
        models_out.append(entry)

        row = {"model": model["label"]}
        seed = setup["seeds"][0]
        for t in setup["templates"]:
            name = f"{t}_{GRID_SUBJECT}_{seed}.png"
            if (folder / name).exists():
                thumb = Image.open(folder / name).convert("RGB")
                thumb.thumbnail((256, 256))
                out_name = f"{slug(model['id'])}_{t}_{GRID_SUBJECT}_{seed}.jpg"
                thumb.save(THUMBS / out_name, quality=82)
                row[t] = f"thumbs/{out_name}"
        grid_rows.append(row)

    ref_thumbs = []
    for f in ref_files[:6]:
        im = Image.open(f).convert("RGB")
        im.thumbnail((256, 256))
        im.save(THUMBS / f"ref_{f.stem}.jpg", quality=82)
        ref_thumbs.append(f"thumbs/ref_{f.stem}.jpg")

    return {
        "artist": setup["artist"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip_model": f"{config.CLIP_MODEL} ({config.CLIP_PRETRAINED})",
        "reference_count": len(ref_files),
        "reference_source": "Art Institute of Chicago Open Access (public domain, CC0)",
        "templates": setup["templates"],
        "subjects": setup["subjects"],
        "seeds": setup["seeds"],
        "near_copy_threshold": config.T_POSSIBLE,
        "models": models_out,
        "grid": {"subject": setup["subjects"][GRID_SUBJECT], "rows": grid_rows},
        "reference_thumbs": ref_thumbs,
        "caption": ("Style lift is how much closer outputs move to the artist's reference works when the "
                    "prompt names the artist. A drop across versions may indicate an opt-out was honored, "
                    "or may reflect other training changes. This is a signal, not proof."),
    }


def main() -> None:
    from core.scan import get_embedder

    embedder = get_embedder()
    results = analyze(embedder.embed)
    (config.AUDIT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    for m in results["models"]:
        print(f"{m['model']}: lift {m['lift']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}]  "
              f"genre lift {m['genre_lift']:+.3f}  near-copy {m['near_copy_rate']:.0%}")


if __name__ == "__main__":
    main()
