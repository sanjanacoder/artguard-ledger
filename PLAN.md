# Development Plan: ArtGuard Ledger

A single Streamlit app where an independent artist uploads a batch of images and gets back:
1. an **exposure report** (per-piece score, label, ranked next steps),
2. a **ZIP of tagged copies** ("Do Not Train" metadata added; originals untouched),
3. a **dated PDF log** (hashes, actions, rights summaries, limitations).

It **measures exposure and documents actions**. It does not guarantee protection, and every screen says so.

---

## 0. Hackathon fit: Banana Hacks 2026

Source: bananahacks.tech (home page, FAQ, prizes, schedule, code of conduct), checked 2026-10-10.

### 0.1 Hard rules (non-negotiable)
| Rule | What we do |
|---|---|
| Hacking window: Fri Oct 9, 8:45 PM EDT to **Sun Oct 11, 11:59 PM AoE (= Mon Oct 12, 7:59 AM EDT)**. No late entries. | Internal code freeze **Sun 6 PM EDT**; submit by **Sun 10 PM EDT** (§6) |
| All work made during the window; **public GitHub repo**; commit timestamps are checked | Create the public repo now and commit early and often. Build the index and the audit generations during the window, and commit the build scripts |
| Theme ("revealed at the opening ceremony") must be addressed | ☐ Confirm the announced theme matches "generative AI, images, or the tools people use to make them" |
| Eligibility: students aged 14–24 currently enrolled; teams of up to 4 | ☐ Confirm for every team member |
| Cite open-source libraries, datasets and pre-trained models in the README | `README.md` → "Credits" section (§11.1) |
| Declare the generative AI tools used | README and submission form: "Built with help from Claude Code (Anthropic)" plus any image models used in the audit |
| Demoable over screen share or at a hosted link; judges may test live demos | Host on Hugging Face Spaces (§4.1), with a **"Load demo batch"** button |

### 0.2 Submission deliverables
- [ ] Project name
- [ ] One-paragraph description
- [ ] Public GitHub link
- [ ] Demo video, **3 minutes or less** (§10)
- [ ] Write-up of **500 words or less** (Devpost-style; §11.2)
- [ ] Live URL (optional, but judges may test it)

### 0.3 Judging rubric (5 criteria, equally weighted, each scored 1–5) and how we score on each
| Criterion | Risk | How we earn the points |
|---|---|---|
| Technical execution & difficulty ("thin API wrappers score lower") | Low | Calibrated CLIP thresholds, a pHash check, a robustness chart, metadata tags read back to verify them, a hash-based audit trail, and the style-lift statistics |
| Innovation & creativity ("generic clones score lower") | Medium | Don't pitch it as "Have I Been Trained plus links". Lead with what's new: one workflow, a **verifiable record**, an honest coverage map, and a **style-lift audit** of generative models |
| **Use of image AI theme** ("if the image AI could be removed without breaking the core experience, the score drops") | **High** | Make image AI the core: (1) the CLIP scan is the centerpiece of the UI and the video, (2) the **style-lift audit** (F9) runs generative image models, (3) optionally, a trained copy-detection head (S1). Tagging, PDF and rights summaries are the supporting features |
| Impact & usefulness | Low | A clear user (independent artists), a real problem, and outputs they can use after the weekend |
| Design, presentation & demo | Medium | A custom Streamlit theme, status chips, a tight 3-minute video, and the stripped-tag moment |

**Prize targets:** top project, People's Choice, and **Most Surprising Output** (via the style-lift audit). Best Fine-tune is only reachable if S1 lands. Best Creative Tool is a weak fit (we protect makers from generative AI rather than giving it to them), so we don't optimize for it.

---

## 1. Scope

### 1.1 MVP (must ship for the demo)
| # | Feature | Definition of done |
|---|---------|--------------------|
| F1 | Batch upload + artist info | JPEG/PNG/WebP upload (≤20 files, ≤10 MB each); optional name, website, "publicly posted?" and "used AI tools?" per piece |
| F2 | Dataset scan | CLIP similarity against a demo index; each piece labeled **Strong / Possible / No match** with the top 3 nearest neighbors shown |
| F3 | Do Not Train tagging | Tagged copy of each file with IPTC/PLUS `DataMining` + usage terms in XMP; tag verified by reading it back |
| F4 | Opt-out & style-protection links | Per-piece "Needs you" actions linking to opt-out registries and style-protection tools |
| F5 | Coverage map | Protection × trainer table with Yes / No / Unknown, each cell sourced and dated |
| F6 | Rights summary | Short questionnaire → rule-based plain-English summary based on US Copyright Office guidance, with a "not legal advice" notice |
| F7 | Exposure score + report | 0–100 score, High/Medium/Lower label, weights shown, ranked next steps marked **Done by app / Needs you / Can't verify** |
| F8 | Downloads | ZIP of tagged copies + PDF log (header, per-piece table with SHA-256, action history, rights summaries, limitations) |
| F9 | **Style-lift audit (lite)** | Precomputed generations from 2 model versions × styled/control prompts; CLIP "style lift" with confidence intervals, shown as a chart in its own tab (§9) |
| F10 | **Scan robustness view** | A chart showing how similarity holds up under crop, resize, recompression and color shift for a seeded image, which demonstrates why CLIP beats exact hashing |
| F11 | **Demo mode + hosting** | A "Load demo batch" button; deployed to Hugging Face Spaces; README with credits and AI-tool disclosure |

### 1.2 Out of scope for the hackathon
- Scanning real, full-size datasets (LAION-5B etc.). The demo uses a sample index, and the UI says so.
- Applying Glaze/Nightshade inside the app. They are separate desktop tools, so we link to them.
- Submitting opt-outs to registries on the artist's behalf. We prepare and link; the artist acts.
- Accounts, persistence, and multi-user. Everything lives in session memory.
- **Cut due to the deadline:** C2PA manifests, independent timestamping (OpenTimestamps), and live image generation inside the hosted app (audit generations are precomputed).

### 1.3 Stretch goals (in priority order, only after the MVP checklist in §5 passes)
1. **S1: Copy-detection head.** Train a small MLP or linear head on pairs of CLIP embeddings (positives = augmented copies, negatives = unrelated images) to replace fixed thresholds. Report precision and recall against thresholds alone. This strengthens technical execution and is the only realistic route to Best Fine-tune.
2. **S2: Before/after score.** Show the current exposure and the projected exposure once the "Needs you" steps are done.
3. **S3: Live audit generation.** A "generate 4 more" button in the audit tab, if sponsor compute credits make inference cheap.

---

## 2. Architecture

```
┌──────────────────────── Streamlit UI (app.py) ───────────────────────┐
│ 1 Upload → 2 Questions → 3 Run scan → 4 Report → 5 Download          │
└───────┬──────────────────────────────────────────────────────────────┘
        │  Piece objects held in st.session_state
        ▼
  core/pipeline.py  ── orchestrates and appends to ActionLog
        ├── core/hashing.py      SHA-256 of original + tagged bytes
        ├── core/scan.py         CLIP embed → cosine top-k vs. index → match label
        ├── core/tagging.py      write/verify XMP "Do Not Train" on copies
        ├── core/coverage.py     load data/coverage.json → table
        ├── core/rights.py       questionnaire answers → summary (rule-based)
        ├── core/scoring.py      signals → score, label, next steps
        ├── core/export_zip.py   zipfile of tagged copies + manifest.json
        └── core/export_pdf.py   ReportLab log
  index/
        ├── build_index.py       offline: images → embeddings.npy + meta.json
        ├── embeddings.npy       float16, L2-normalized
        └── meta.json            id, source dataset, caption, url, license
  audit/   (offline generate → analyze; app only renders results.json)
        ├── generate.py          image models × styled/control prompts × seeds
        └── analyze.py           CLIP style lift + bootstrap CI → results.json
```

**Design rules**
- `core/` has **no Streamlit imports**, so every module can be unit-tested and reused from a CLI.
- Originals are never written to. All writes happen on `bytes` copies, and an original's hash is checked again at export time to prove it is unchanged.
- Every step appends a timestamped `Action` to the log. The report and PDF are rendered from that log, so what we show is exactly what happened.

### 2.1 Data model (`core/models.py`)
```python
@dataclass
class Action:
    ts_utc: str                 # ISO-8601
    piece_id: str | None        # None = batch-level
    step: str                   # "hash", "scan", "tag", "verify_tag", "export"...
    status: Literal["done_by_app", "needs_you", "cant_verify", "failed"]
    detail: str

@dataclass
class MatchResult:
    label: Literal["strong", "possible", "none"]
    top_score: float
    neighbors: list[dict]       # [{id, score, source, caption, thumb_url}]

@dataclass
class Piece:
    id: str                     # short uuid
    filename: str
    mime: str
    original_bytes: bytes
    sha256_original: str
    publicly_posted: bool = True
    opted_out: bool = False     # artist self-reports
    style_protected: bool = False  # artist self-reports (Glaze etc.)
    ai_answers: dict = field(default_factory=dict)
    match: MatchResult | None = None
    tagged_bytes: bytes | None = None
    sha256_tagged: str | None = None
    tag_verified: bool = False
    rights_summary: str | None = None
    score: int | None = None
    label: str | None = None
    next_steps: list[dict] = field(default_factory=list)
```

---

## 3. Component specs

### 3.1 Upload (F1)
- `st.file_uploader(accept_multiple_files=True, type=["jpg","jpeg","png","webp"])`.
- Validate each file by opening it with Pillow (`Image.verify()`). Reject files that don't open, cap the count and size, and strip nothing.
- Per-piece questions go in one `st.data_editor` table (filename | publicly posted | already opted out | Glazed/Nightshaded | AI used?) so a batch of 20 doesn't mean 20 forms.
- Hash immediately on upload (`hashlib.sha256(bytes).hexdigest()`) and log a `hash` action.

### 3.2 Dataset scan (F2)

**Model:** `open_clip` ViT-B-32 (`laion2b_s34b_b79k`). It runs on CPU at roughly 0.1–0.3 s per image and is cached with `@st.cache_resource`. If the hardware is weak, swap in the smaller `ViT-B-32-quickgelu`. No GPU is needed.

**Index (built offline by `index/build_index.py`):**
- Use 5–20k images from openly licensed sources (for example a WikiArt or public-domain art subset from Hugging Face, plus some general photos). Record source and license in `meta.json`.
- **Seed the demo:** add 3–4 of the demo artist's images (images *we* own or made for the demo) as transformed near-duplicates: resized, JPEG-recompressed, cropped 10%, slightly color-shifted. This mimics how scraped copies look, and it gives a reproducible Strong match in the live demo.
- Store L2-normalized embeddings as `float16` `.npy`. At this size, brute-force `embeddings @ query` with NumPy takes milliseconds, so FAISS is optional (keep `faiss-cpu` as a drop-in if the index grows past ~100k).

**Matching:**
```
sim = index_emb @ query_emb          # cosine, since both are normalized
top_k = argsort(sim)[-5:]
strong   if top ≥ T_STRONG
possible if top ≥ T_POSSIBLE
none     otherwise
```
- **Calibrate the thresholds** in a notebook (`notebooks/calibrate.ipynb`): compute similarities for known near-duplicate pairs (the seeded transforms) and for unrelated pairs. Pick `T_STRONG` above about 99% of unrelated pairs, and `T_POSSIBLE` to catch heavy crops or recolors. Expect roughly 0.90 / 0.80 for ViT-B-32 image-to-image, but **measure, don't guess**. Store the values in `config.py`.
- Optional precision boost: run a perceptual hash (`imagehash.phash`) on the top neighbor. A Hamming distance ≤ 8 together with CLIP ≥ `T_POSSIBLE` upgrades the piece to Strong. This is cheap and cuts false "strong" labels on images that are merely similar in style.
- The UI shows neighbor thumbnails with their scores, so the artist can judge matches themselves.
- Every scan result includes the line: *"Searched: demo index (N images, sources …). Not searched: everything else."*

**Robustness view (F10):** during calibration, save the similarity of a seeded image against its own transforms (crop at 10/25/40%, resize 50/25%, JPEG quality 90/60/30, hue shift, grayscale, screenshot-style border) and against the nearest unrelated image. Plot it as a bar chart with the Strong/Possible thresholds drawn as lines. Precompute it into `index/robustness.json` and show it in a "How the scan works" tab. This is the visual that tells judges the scan is real computer vision and not just a file-hash lookup.

### 3.3 Do Not Train tagging (F3)
**What we write** (XMP, embedded in the copy):
- `XMP-plus:DataMining` = `DMI-PROHIBITED-AIMLTRAINING`. This is the IPTC/PLUS "Data Mining" property added in IPTC Photo Metadata Standard 2023.1.
- `XMP-plus:OtherConstraints` = a human-readable sentence ("Use of this work for AI/ML training is not permitted by the creator.").
- `XMP-xmpRights:UsageTerms` and `XMP-dc:Rights` = the same text plus the artist name and website, if given.
- `XMP-dc:Creator` = the artist name, if given.

**How we write it**
- **Primary:** ExifTool via `pyexiftool` (or `subprocess`). It handles JPEG, PNG and WebP correctly and knows the PLUS namespace. Write to a temp copy, then read the bytes back.
- **Fallback, if ExifTool isn't installed:** hand-build an XMP packet and embed it with Pillow (a JPEG APP1 segment, or an `iTXt` chunk with key `XML:com.adobe.xmp` for PNG). Test the fallback early. If it's flaky, document ExifTool as a required dependency (`brew install exiftool`) and keep the fallback for PNG only.
- **Verify:** read the tag back from the tagged bytes. If the value matches, set `tag_verified=True` and log `done_by_app`. Otherwise log `failed` and surface it in the UI.
- Hash the tagged bytes, and recompute the original's hash to confirm it hasn't changed.

*(C2PA manifests are cut for this hackathon. List them under "Future work" in the write-up.)*

### 3.4 Opt-out & style-protection links (F4)
Keep the links in `data/resources.json` so they can be edited without code changes. Each entry has `name`, `url`, `what_it_does`, `who_honors_it` (link to coverage), and `last_checked`. Starting candidates, **each to be verified by the team before the demo**:
- Opt-out / consent registries: Spawning (Have I Been Trained / Do Not Train registry).
- Platform-level settings: the artist's own hosting platforms' AI/scraping toggles, plus `robots.txt` guidance for their own website (blocking known AI crawler user-agents).
- Style protection: Glaze and Nightshade (University of Chicago).

These always show up in the report as **Needs you** actions. If the artist ticked "already opted out" or "already Glazed", show them as **Can't verify** (self-reported), never as Done.

### 3.5 Coverage map (F5)
- `data/coverage.json` holds rows = protections (DNT metadata, registry opt-out, robots.txt / crawler block, Glaze/Nightshade, C2PA assertion) and columns = trainers/crawlers (a handful of major model developers and the main open crawl/dataset sources).
- Every cell is `{value: "yes"|"no"|"unknown", source_url, note, as_of}`. **Default every cell to `unknown`.** Change a cell to yes/no only when there is a public statement or document to cite. Unknown is the honest default, and it supports the app's whole argument.
- Render it with `st.dataframe` and colored cells. A tooltip or expander shows the source for each cell.
- One person owns this file and fills it during hours 4–14 (see §6). It's research, not code.

### 3.6 Rights summary (F6)
**Questionnaire** (per piece, or "apply to all"):
1. How was this piece made? *Entirely by hand/traditional digital* · *I used AI tools for minor edits (cleanup, upscaling, fill)* · *I generated parts with AI and combined or edited them substantially* · *Mostly AI-generated from my prompts*
2. Did you substantially modify AI output by hand? yes/no/n.a.
3. Is the piece a selection or arrangement of AI-generated elements you chose? yes/no/n.a.

**Engine:** a deterministic rule table (`data/rights_rules.yaml`) maps answer combinations to one of about five template summaries. **No LLM**: the output must be predictable and reviewable. Every template:
- states the likely status in 2–4 plain sentences (e.g. human-authored elements are protectable; material generated by AI alone is not, and prompts alone are generally not enough; human selection, arrangement or modification can be protectable),
- notes the registration-disclosure point (AI-generated material beyond a de minimis amount should be disclosed or excluded when registering),
- cites the source: the US Copyright Office's March 2023 registration guidance on works containing AI-generated material, and its *Copyright and Artificial Intelligence, Part 2: Copyrightability* report (January 2025),
- ends with: **"This is general information, not legal advice. Consult an attorney for your situation."**

Before the demo, someone re-reads the templates against the two USCO sources and quotes or links them exactly.

### 3.7 Exposure score (F7)
Weights live in `config.py` and are shown in the UI ("How this score works"):

| Signal | Points |
|---|---|
| Strong dataset match | +50 |
| Possible dataset match (mutually exclusive with strong) | +25 |
| No Do Not Train tag | +15 |
| No opt-out registration | +15 |
| Publicly posted | +10 |
| No style protection | +10 |

The maximum is 50+15+15+10+10 = **100**, so no clamping is needed. Assert this in a test.

**Labels** (also a design choice, also shown): High ≥ 60 · Medium 30–59 · Lower < 30.

**What counts as "tagged":** the copies we produce are tagged, but the versions already online are not. So the score is computed for the **currently posted** state, with tag = missing. The stretch "after" score (S2) shows what it would be if the artist re-uploaded the tagged copies and completed the Needs-you items. This keeps the score honest.

**Next steps** are generated from the signals that contributed points and ranked by those points, e.g.:
1. (+50 strong match) "Review the matched copies. Consider a takedown or opt-out request for this image." — *Needs you*
2. (+15 no tag) "Replace the posted file with the tagged copy." — *Done by app* (copy created) + *Needs you* (re-upload)
3. (+15 no opt-out) "Register with <registry>." — *Needs you* (or *Can't verify* if self-reported)
4. (+10 no style protection) "Consider Glaze/Nightshade before posting new work." — *Needs you*
5. "Whether trainers honor these signals" — always *Can't verify*

### 3.8 Report screen
- Summary row with `st.metric`: pieces scanned, High / Medium / Lower counts, strong matches, tags verified.
- Main table (`st.dataframe` with `ImageColumn` fed by data-URI thumbnails): thumb | filename | match | score | label | # Needs-you.
- Below it, one `st.expander` per piece: neighbors with scores, a score breakdown (bar of contributions), next steps with status chips, the rights summary, and both hashes.
- A persistent "Limits" box (§8) on the report and download screens.

### 3.9 Downloads (F8)
**ZIP** (`zipfile`, in memory via `io.BytesIO`):
```
tagged/<original-name>        # tagged copies, same filenames
manifest.json                 # per file: sha256_original, sha256_tagged, tags written, verified
README.txt                    # what was done, how to check the tag (exiftool command), limits
```

**PDF** (ReportLab Platypus):
1. Header: artist name, website, generated-at timestamp (UTC and local), app version, and index description (name, size, date built).
2. Summary counts.
3. Per-piece table: thumbnail, filename, SHA-256 (original), SHA-256 (tagged), match label + top score, score/label. Use a monospace font for hashes and wrap them.
4. Action history: the complete `ActionLog` in chronological order.
5. Rights summaries per piece, with the disclaimer.
6. Coverage map snapshot, with its as-of date.
7. Limitations section (verbatim from §8).
8. Footer on every page: "Record generated by ArtGuard Ledger. Not a legally certified timestamp." and page numbers.

After building the PDF, compute its SHA-256 and show it on screen, so the artist can store it separately.

---

## 4. Repository layout
```
artguard-ledger/
├── app.py                    # Streamlit entry; UI only
├── config.py                 # thresholds, weights, labels, limits text
├── core/
│   ├── models.py  pipeline.py  hashing.py  scan.py  tagging.py
│   ├── coverage.py  rights.py  scoring.py  export_zip.py  export_pdf.py
├── data/
│   ├── coverage.json  resources.json  rights_rules.yaml
├── index/
│   ├── build_index.py  embeddings.npy  meta.json  thumbs/
│   ├── robustness.json       # F10, precomputed
├── audit/
│   ├── prompts.yaml          # subjects, styled/control templates, seeds
│   ├── generate.py           # offline: run models → audit/outputs/<model>/...
│   ├── analyze.py            # CLIP style lift + bootstrap CI → results.json
│   ├── reference/            # reference-artist images (public domain)
│   └── results.json          # what the app renders (outputs/ is .gitignored if large)
├── demo/
│   ├── artist_samples/       # images we own, for the live demo
│   └── script.md             # demo video talk track
├── notebooks/calibrate.ipynb
├── tests/
│   ├── test_scoring.py  test_tagging.py  test_hashing.py
│   ├── test_rights.py   test_scan.py     test_exports.py  test_audit.py
├── .streamlit/config.toml    # custom theme
├── packages.txt              # apt deps for hosting: libimage-exiftool-perl
├── requirements.txt
├── WRITEUP.md                # ≤500 words, pasted into the submission
└── README.md                 # quickstart, credits, AI-tool disclosure, limits
```

**Dependencies:** `streamlit`, `open_clip_torch`, `torch` (CPU), `pillow`, `numpy`, `imagehash`, `pyexiftool` (+ the ExifTool binary), `reportlab`, `pyyaml`, `pytest`. Offline only (audit generation): `diffusers` + GPU from sponsor compute credits, **or** a hosted inference API. Optional: `faiss-cpu`.

### 4.1 Hosting
- **Hugging Face Spaces (Streamlit SDK, free CPU tier)** is the primary option. Its RAM is enough for ViT-B-32, and `packages.txt` installs ExifTool through apt.
- Streamlit Community Cloud is the fallback. Its memory is tighter, so test that the CLIP model loads there early.
- Commit the precomputed `embeddings.npy`, `meta.json`, thumbnails, `robustness.json` and `audit/results.json` (with audit images resized to small thumbnails), so the hosted app never builds the index or generates images.
- Bake the CLIP weights download into a warm-up call at startup, so judges don't wait on the first scan.
- Deploy the walking skeleton by **Sat evening** so hosting problems surface early, not on Sunday night.

---

## 5. Implementation order

Each step is independently demoable. Build the thin end-to-end path first, then deepen it.

| Step | Work | Output |
|---|---|---|
| 0 | **Public GitHub repo + first commit** (this plan), skeleton, `requirements.txt`, `models.py`, `config.py`, an upload page that lists hashes | A clickable shell; the commit-timestamp trail starts |
| 1 | **Walking skeleton:** a stub scan (random label), stub tag, real scoring, ZIP, and a minimal PDF, all wired end to end, **deployed to HF Spaces** | Full flow works with fake data at a public URL |
| 2 | `build_index.py` + seeded demo images, then the real `scan.py` + calibration + `robustness.json` | Real Strong/Possible/None, plus the F10 chart |
| 3 | Real `tagging.py` with ExifTool + read-back verification + PNG fallback | Verified tagged copies |
| 4 | **Audit:** `prompts.yaml` → `generate.py` (runs in parallel on GPU or API) → `analyze.py` → audit tab | F9 chart with CIs |
| 5 | `rights.py` + rules YAML + questionnaire UI | Rights summaries |
| 6 | `coverage.json` research + table UI; `resources.json` links | Coverage map and next steps |
| 7 | Full report UI (metrics, table, expanders, score breakdown), custom theme, demo-batch button | Polished report |
| 8 | Full PDF layout (thumbnails, tables, log, limits, footer) | Final PDF |
| 9 | Tests, edge states, copy review, README credits, write-up, **record video** | Submittable |
| 10 | Stretch goals (§1.3) | Bonus |

**MVP checklist (must pass before any stretch work):** F1–F11 work on the hosted URL · tests are green · the README credits and AI disclosure are done · the video is recorded · the write-up is ≤500 words.

---

## 6. Timeline and team split (Sat Oct 10 → Sun Oct 11, all times EDT, 3–4 people)

Real deadline: **Mon Oct 12, 7:59 AM EDT** (Sun 11:59 PM AoE). We plan to **submit by Sun 10 PM EDT** and treat the remaining hours as buffer only.

| When (EDT) | **A: Scan & ML** | **B: Tagging, exports, audit** | **C: UI & hosting** | **D: Research & content** |
|---|---|---|---|---|
| Sat, now → +2 h | Repo + skeleton + models/config | Hashing, ZIP stub | Upload page, session state | Resource links; limitations text; pick audit models + reference artist |
| Sat, +2 → +6 h | Dataset subset, `build_index.py` | ExifTool tagging + verify | Walking skeleton → **deploy to HF Spaces** | `coverage.json` (sources!); audit `prompts.yaml` |
| Sat evening | Seed transforms, calibrate, real scan, `robustness.json` | **Kick off `generate.py`** (runs in background), then PNG/WebP fallback | Questions `data_editor`, report table, theme | Rights questionnaire + rule templates |
| Sat night | pHash check, neighbor thumbs | `analyze.py` → `results.json` | Expanders, score breakdown, audit tab | Verify USCO wording; demo images |
| Sun morning | Scoring + scan tests | ReportLab PDF (full layout) | Robustness chart, download screen, Limits box | README credits; write-up draft |
| Sun midday | **Integration + bug bash on the hosted URL (everyone)** | | | |
| **Sun 6 PM** | **Code freeze.** Only bug fixes after this | | | |
| Sun 6–9 PM | S1 copy-detection head *only if MVP is green* | Final PDF check | Record video (2–3 takes) | Finalize write-up + description |
| **Sun 10 PM** | **Submit.** Confirm the repo is public and the latest commit is in | | | |

With fewer people, merge A+B and C+D. If the audit generations aren't done by Sun morning, ship whatever pairs have finished. One model version with styled vs. control prompts still tells the story.

---

## 7. Testing

**Unit (pytest)**
- `scoring`: max = 100; strong and possible are mutually exclusive; each signal contributes the right points; label boundaries at 29/30/59/60.
- `hashing`: known vector; recomputing the original's hash after tagging gives the same value (originals untouched).
- `tagging`: for JPEG, PNG and WebP fixtures, the tag reads back as `DMI-PROHIBITED-AIMLTRAINING`; the image still decodes and its pixels are unchanged (compare decoded arrays).
- `scan`: a seeded transform → strong; an unrelated image → none; the result is deterministic.
- `rights`: every answer combination maps to a template; every template contains the disclaimer.
- `exports`: the ZIP contains every file plus manifest; the hashes in manifest and PDF match the computed ones; the PDF opens (parse with `pypdf`).
- `audit`: the lift is ~0 when styled and control are the same image set (sanity check); the bootstrap is deterministic with a fixed seed; `results.json` matches the schema.

**Manual**
- Open a tagged copy in ExifTool or Photoshop/Lightroom and confirm the data-mining field shows.
- Upload a tagged copy to a social platform, download it again, and show the tag is stripped. Put this result in the demo: it is the strongest honest moment of the pitch.
- Run 20 images end to end in under about 60 s on a laptop CPU.

**Edge cases:** CMYK JPEGs, PNGs with alpha, huge images (downscale for CLIP only, never for the copy), duplicate filenames in a batch, a non-image file renamed to `.jpg`, and an empty name/website.

---

## 8. Limitations text (shown in the UI and printed verbatim in the PDF)
- Metadata tags can be removed when images are uploaded to many platforms, edited, or screenshotted.
- The dataset scan covers only the sources searched. This demo searches a sample index of N images and does not search real training datasets in full.
- A "no match" result does not mean the work was not collected.
- Opt-outs and "Do Not Train" signals work only if AI developers choose to honor them. The app cannot verify whether they do.
- Self-reported items (opt-out, style protection) are recorded as stated and not verified.
- This log is a record generated by the app. It is not a legally certified timestamp.
- The rights summary is general information based on US Copyright Office guidance. It is not legal advice.

---

## 9. Style-lift audit, lite (F9, in scope)

This is the feature that puts generative image models at the center of the app (rubric criterion 3) and the most likely route to "Most Surprising Output".

**Question it answers:** "When a model is asked for an artist's style, do its outputs actually move toward that artist's work, and does that change across model versions?"

**Demo subject:** our demo artist isn't known to any model, so for the demo we audit a **public-domain historical artist** as a positive control (e.g. Hokusai or Van Gogh, with reference images from public-domain collections). The UI states that a real user would audit their **own** name only.

**Setup (`audit/prompts.yaml`):**
- 6–8 neutral subjects ("a harbor at dawn", "a cat on a windowsill", …).
- Styled template: "{subject}, in the style of {artist}". Control template: "{subject}, painting".
- 2 model versions (e.g. an older and a newer open Stable Diffusion checkpoint, or whatever is available through sponsor compute credits or a hosted API). Use fixed seeds; K = 4 seeds per subject per template, so 6–8 subjects × 2 templates × 4 seeds × 2 models ≈ 100–130 images.
- Record exact model IDs, sampler, steps and seeds in `results.json`.

**Generation (`audit/generate.py`, offline):** run it on a GPU (sponsor credits or a Colab-style notebook) or through a hosted inference API. Start it **Saturday evening** in the background. Save outputs at 512 px and commit downscaled thumbnails.

**Analysis (`audit/analyze.py`):**
- Reference centroid = mean CLIP embedding of the artist's reference images (normalized).
- For each image, `s = cos(embedding, centroid)`.
- **Style lift** per model = mean(s, styled) − mean(s, control), with a 95% bootstrap CI (1,000 resamples, resampling subjects, fixed seed).
- Also report how many styled images have their nearest reference image above `T_POSSIBLE` ("near-copy rate"). This is often the surprising number.

**UI (audit tab):** a bar chart of lift per model with CI whiskers, a grid of styled vs. control samples for one subject, and the caption: *"A drop in lift across versions **may** indicate an opt-out was honored, or may reflect other training changes. This is a signal, not proof."* The audit's prompts, seeds and model IDs are written into the PDF if the user includes the audit.

**Fallback:** if generation fails entirely, ship the tab with one model and styled vs. control only. The lift alone still demonstrates the method.

---

## 10. Demo video script (≤3 minutes, recorded)
1. **Problem (15 s):** "Separate tools, no visibility, no record."
2. **Upload (15 s):** click "Load demo batch" (5 pieces, one marked AI-assisted).
3. **Scan (40 s):** one Strong match (the seeded image) with its neighbor shown side by side; others Possible or None. Cut to the **robustness chart**: "it still finds a 40% crop recompressed at quality 30." Point out the "searched only the demo index" line.
4. **Report (30 s):** scores and labels, open one expander: score breakdown, next steps with Done/Needs you/Can't verify, and the rights summary for the AI-assisted piece.
5. **Style-lift audit (35 s):** the lift chart across two model versions, plus the styled vs. control grid. "Signal, not proof."
6. **Coverage map (10 s):** "Most of this is Unknown, and that's the point."
7. **Downloads (25 s):** open the ZIP, run `exiftool -XMP-plus:DataMining` on a copy, then show the tag gone after a social-media round trip. Open the PDF: hashes and the limits page.
8. **Close (10 s):** impact and the live URL.

Record 2–3 takes, using the hosted app if it's stable or localhost otherwise. Add captions, and keep the final cut **under 3:00** (aim for 2:50).

---

## 11. Submission materials

### 11.1 README.md
- A one-paragraph pitch and a link to the live demo.
- Quickstart: `pip install -r requirements.txt && brew install exiftool && streamlit run app.py`.
- How it works: scan, tag, score, record, audit (with the weights and thresholds).
- **Credits** (required): Streamlit, OpenCLIP and the CLIP weights used, PyTorch, Pillow, NumPy, imagehash, ExifTool / PyExifTool, ReportLab, PyYAML, the image models used in the audit, every dataset in the index with its license, and the public-domain sources for the reference artist.
- **AI tools used** (required): "Planning and code assisted by Claude Code (Anthropic)", plus the generative image models used in the audit.
- Limitations (§8) and future work (C2PA, real dataset coverage, timestamping).

### 11.2 Write-up (`WRITEUP.md`, ≤500 words)
Inspiration (~60 words) · What it does (~100) · How we built it (~120, focused on CLIP calibration, robustness, tag verification and style lift) · Challenges (~60) · What we learned / honest limits (~80) · What's next (~50). Check the count with `wc -w`.

### 11.3 One-paragraph description
> *ArtGuard Ledger* gives independent artists one place to check whether their work shows up in AI training data, apply "Do Not Train" protections, and keep a dated record. It uses CLIP image similarity to find altered copies in a searchable index, audits whether generative models drift toward an artist's style, tags copies with IPTC data-mining metadata, explains likely copyright status in plain English, and exports a hashed PDF log. It measures and documents exposure; it doesn't promise protection.

---

## 12. Risks & mitigations
| Risk | Mitigation |
|---|---|
| **Missing the deadline** | Code freeze Sun 6 PM EDT, submit Sun 10 PM EDT, video recorded before stretch work |
| Commit-timestamp check fails (work done outside window or not pushed) | Push the first commit now and keep pushing; don't paste in code written before Fri 8:45 PM EDT |
| Audit generation is slow or the GPU isn't available | Start Saturday evening, keep K small, use a hosted API as backup, ship the one-model fallback |
| Hosting fails (RAM, ExifTool, cold start) | Deploy the skeleton Saturday; keep Streamlit Cloud as a fallback; the video shows localhost if needed |
| CLIP download or slow CPU | Warm up at startup, precompute the index, keep the batch ≤20 |
| ExifTool missing on the demo machine | Install it and verify on day 1; keep the Pillow fallback; tests cover both |
| Thresholds produce false Strong matches | Calibrate on held-out pairs, add the pHash check, show neighbors so the user can judge |
| Coverage map claims something wrong | Default to Unknown; every yes/no needs a source URL and as-of date |
| Rights summary sounds like legal advice | Fixed templates, cited sources, disclaimer on every summary and in the PDF |
| Dataset licensing for the demo index | Use only openly licensed or public-domain images and record license in `meta.json` |
| Privacy of uploads | Process in memory only, never write uploads to disk outside temp files deleted after tagging, and say so in the UI |
| Scope creep | Code freeze Sun 6 PM EDT; stretch goals only after the MVP checklist (§5) passes |

---

## 13. Judging-oriented polish (if time allows)
- A "How this works" page: the score weights, thresholds, robustness chart, and what was searched.
- An accessible design: color plus text labels for High/Medium/Lower, and alt text on thumbnails.
- A custom `.streamlit/config.toml` theme (banana-yellow accent) and a simple logo, for a consistent look across the app, the video and the README.
