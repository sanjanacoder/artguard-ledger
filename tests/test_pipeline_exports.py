"""End-to-end pipeline and exports, using a tiny fake index and embedder (no CLIP download)."""
import io
import json
import zipfile

import numpy as np
import pytest
from pypdf import PdfReader

from core import pipeline
from core.export_pdf import build_pdf
from core.export_zip import build_zip
from core.scan import ClipIndex, classify
from tests.conftest import make_image


class FakeEmbedder:
    """Embeds by mean color, so identical images match exactly."""
    name = "fake"

    def embed(self, images):
        out = []
        for im in images:
            v = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32).ravel() + 1
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def batch_and_index():
    a, b = make_image("JPEG"), make_image("PNG", "RGBA", size=(40, 40))
    emb = FakeEmbedder()
    from core.scan import load_rgb
    vec = emb.embed([load_rgb(a)])[0]
    other = np.ones_like(vec) / np.sqrt(len(vec))
    index = ClipIndex(np.stack([vec, other]),
                      [{"id": "copy", "title": "copy of a"}, {"id": "other", "title": "flat"}],
                      {"name": "test index", "size": 2, "built_at": "now", "sources": [{"name": "t", "count": 2}]})
    batch = pipeline.Batch(artist="Jane", website="https://jane.example")
    pipeline.ingest(batch, "a.jpg", a)
    pipeline.ingest(batch, "b.png", b)
    pipeline.ingest(batch, "a.jpg", a)  # duplicate name gets renamed
    return batch, index, emb


def test_full_run(batch_and_index):
    batch, index, emb = batch_and_index
    pipeline.run(batch, index, emb, tag_engine="builtin")
    names = [p.filename for p in batch.pieces]
    assert names == ["a.jpg", "b.png", "a-2.jpg"]
    a = batch.pieces[0]
    assert a.match.label == "strong" and a.match.top_score > 0.999
    assert all(p.tag_verified for p in batch.pieces)
    assert all(p.score is not None and p.label for p in batch.pieces)
    assert pipeline.verify_originals(batch)
    steps = {x.step for x in batch.log.actions}
    assert {"hash", "scan", "tag", "verify_tag", "rights", "score", "verify_original"} <= steps


def test_zip_matches_hashes(batch_and_index):
    batch, index, emb = batch_and_index
    pipeline.run(batch, index, emb, tag_engine="builtin")
    zf = zipfile.ZipFile(io.BytesIO(build_zip(batch)))
    manifest = json.loads(zf.read("manifest.json"))
    assert len(manifest["files"]) == 3
    from core.hashing import sha256_bytes
    for entry, p in zip(manifest["files"], batch.pieces):
        assert sha256_bytes(zf.read(entry["path"])) == entry["sha256_tagged"] == p.sha256_tagged
        assert entry["sha256_original"] == p.sha256_original
    assert "README.txt" in zf.namelist()


def test_pdf_contains_hashes_and_limits(batch_and_index):
    batch, index, emb = batch_and_index
    pipeline.run(batch, index, emb, tag_engine="builtin")
    audit = {"artist": "X", "caption": "c", "models": [
        {"model": "m", "lift": 0.05, "ci_low": 0.01, "ci_high": 0.09, "n_styled": 4}]}
    reader = PdfReader(io.BytesIO(build_pdf(batch, audit)))
    text = "".join(page.extract_text() for page in reader.pages).replace("\n", "")
    for p in batch.pieces:
        assert p.sha256_original[:32] in text
    assert "Limitations" in text and "not legal advice" in text


def test_rejects_non_images():
    batch = pipeline.Batch()
    with pytest.raises(pipeline.RejectedFile):
        pipeline.ingest(batch, "fake.jpg", b"not an image")


def test_rejects_gif():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "GIF")
    with pytest.raises(pipeline.RejectedFile):
        pipeline.ingest(pipeline.Batch(), "x.gif", buf.getvalue())


@pytest.mark.parametrize("score,dist,expected", [
    (0.99, None, ("strong", False)), (0.0, 0, ("none", False)),
    (None, 2, ("strong", True)), (None, 30, ("possible", False)),
])
def test_classify(score, dist, expected):
    import config
    if score is None:
        score = (config.T_STRONG + config.T_POSSIBLE) / 2
    assert classify(score, dist) == expected
