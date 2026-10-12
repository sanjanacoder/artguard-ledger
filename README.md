# 🍌 ArtGuard Ledger

**See what's been taken. Document what you've done.**

Independent artists have little visibility into whether their work has been collected for AI training, and the protections available are spread across separate tools. ArtGuard Ledger is a single web app: upload a batch of images and it

1. **scans** each one against a searchable image index with CLIP image similarity (strong / possible / no match),
2. **tags** copies with IPTC "Do Not Train" metadata (originals untouched) and verifies every tag,
3. **scores** exposure from 0 to 100 using published weights, with ranked next steps marked *Done by app*, *Needs you* or *Can't verify*,
4. **explains** each piece's likely copyright status in plain English (US Copyright Office guidance, not legal advice),
5. **audits** whether generative image models move toward an artist's style when prompted with their name, and
6. **exports** a ZIP of tagged copies and a dated PDF log with SHA-256 hashes.

It measures exposure and documents actions. **It does not guarantee protection.**

Built for [Banana Hacks 2026](https://www.bananahacks.tech/).

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install exiftool          # optional; a built-in pure-Python writer is used without it
streamlit run app.py
```

Click **Load demo batch** to try it with five sample pieces. The search index, calibration and audit results are committed, so the app needs no GPU and no downloads beyond the CLIP weights (about 600 MB, fetched on first run).

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## How it works

| Step | What happens | Code |
|---|---|---|
| Scan | OpenCLIP ViT-B-32 image embeddings, cosine similarity against the index. Thresholds are **measured** on edited copies vs. unrelated held-out images (`index/calibrate.py`). A perceptual hash can upgrade a possible match to strong. | `core/scan.py` |
| Tag | Writes `plus:DataMining = http://ns.useplus.org/ldf/vocab/DMI-PROHIBITED-AIMLTRAINING` (IPTC Photo Metadata 2023.1) plus a plain-text notice, via ExifTool or a built-in XMP writer for JPEG, PNG and WebP. Pixels are never re-encoded. Every tag is read back. | `core/tagging.py` |
| Score | Strong match +50, possible +25, no tag on the posted file +15, no opt-out +15, publicly posted +10, no style protection +10. High ≥ 60, Medium 30–59, Lower < 30. | `core/scoring.py`, `config.py` |
| Rights | Fixed rule table from questionnaire answers. No LLM, so the output is predictable and reviewable. | `core/rights.py`, `data/rights_rules.yaml` |
| Coverage map | Protection × AI developer table. Every cell defaults to **unknown**; yes/no needs a cited source. | `core/coverage.py`, `data/coverage.json` |
| Style audit | Same subjects prompted "in the style of {artist}", as a generic painting, and as the artist's genre without the name. Style lift = CLIP similarity to the artist's reference works, styled minus control, with paired bootstrap CIs. | `audit/` |
| Record | Every step is appended to a timestamped action log; the report, ZIP manifest and PDF are generated from it. | `core/pipeline.py`, `core/export_*.py` |

### Rebuilding the data

```bash
python demo/make_demo_art.py                 # demo artist's portfolio (procedural, ours)
python index/build_index.py --count 3000     # download index images + embed
python index/calibrate.py --write            # measure thresholds, robustness chart
pip install -r requirements-audit.txt
python audit/fetch_reference.py              # public-domain reference works
python audit/generate.py                     # needs a GPU or Apple Silicon
python audit/analyze.py                      # writes audit/results.json
```

## Limitations

- Metadata tags can be removed when images are uploaded to many platforms, edited, or screenshotted.
- The scan covers only the demo index (about 2,900 public-domain artworks plus simulated copies). It does not search real training datasets.
- "No match" does not mean the work was not collected.
- Opt-outs and Do Not Train signals work only if AI developers choose to honor them.
- The log is a record, not a legally certified timestamp.
- The rights summary is general information, not legal advice.
- The style audit is a signal, not proof.

## Credits

**Libraries:** [Streamlit](https://streamlit.io), [OpenCLIP](https://github.com/mlfoundations/open_clip), [PyTorch](https://pytorch.org), [Pillow](https://python-pillow.org), [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [Altair](https://altair-viz.github.io), [ImageHash](https://github.com/JohannesBuchner/imagehash), [ExifTool](https://exiftool.org) by Phil Harvey, [ReportLab](https://www.reportlab.com), [PyYAML](https://pyyaml.org), [Requests](https://requests.readthedocs.io), [🤗 Diffusers](https://github.com/huggingface/diffusers) and [Transformers](https://github.com/huggingface/transformers) (audit only), [pytest](https://pytest.org) and [pypdf](https://github.com/py-pdf/pypdf) (tests).

**Pre-trained models:**
- CLIP ViT-B-32 trained on LAION-2B (`laion2b_s34b_b79k`) via OpenCLIP: dataset scan and audit analysis.
- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (CreativeML OpenRAIL-M) and [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) (Stability AI): style audit image generation, run offline.

**Data:**
- Index images and audit reference works: [Art Institute of Chicago API](https://api.artic.edu/docs/), public-domain works under CC0.
- Demo artist portfolio: generated procedurally by `demo/make_demo_art.py` for this project.
- Rights summaries are based on the US Copyright Office's [registration guidance for works containing AI-generated material](https://www.copyright.gov/ai/ai_policy_guidance.pdf) (March 2023) and [Copyright and Artificial Intelligence, Part 2: Copyrightability](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf) (January 2025).
- Metadata vocabulary: [IPTC Photo Metadata Standard](https://iptc.org/standards/photo-metadata/) (PLUS Data Mining property).

**AI tools used:** Planning and code were written with help from **Claude Code** (Anthropic). Stable Diffusion 1.5 and SD-Turbo generated the style-audit images.
