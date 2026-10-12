"""ArtGuard Ledger: Streamlit UI. All logic lives in core/."""
from __future__ import annotations

import base64
import json

import altair as alt
import pandas as pd
import streamlit as st

import config
from core import coverage, pipeline, rights
from core.export_pdf import build_pdf
from core.export_zip import build_zip
from core.hashing import sha256_bytes
from core.models import STATUS_TEXT
from core.scan import ClipIndex, get_embedder

st.set_page_config(page_title=config.APP_NAME, page_icon="🍌", layout="wide")

STATUS_BADGE = {"done_by_app": "green", "needs_you": "orange", "cant_verify": "gray", "failed": "red"}
LABEL_BADGE = {"High": "red", "Medium": "orange", "Lower": "green"}
MATCH_TEXT = {"strong": "Strong match", "possible": "Possible match", "none": "No match"}
Q = rights.questions()
METHOD_OPTIONS = Q["method"]["options"]
YN_OPTIONS = Q["modified"]["options"]

# Demo batch: which answers to pre-fill for each sample piece.
DEMO_ANSWERS = {
    "tide_circles.jpg": {"posted": True, "opted_out": False, "glazed": False, "method": "human"},
    "flow_field.png": {"posted": True, "opted_out": False, "glazed": False, "method": "ai_minor"},
    "bauhaus_blocks.jpg": {"posted": False, "opted_out": True, "glazed": False, "method": "human"},
    "paper_mountains.jpg": {"posted": True, "opted_out": False, "glazed": True, "method": "human"},
    "mosaic_sun.webp": {"posted": True, "opted_out": False, "glazed": False, "method": "ai_parts",
                        "modified": "yes", "arranged": "yes"},
}


@st.cache_resource(show_spinner="Loading the CLIP image model…")
def load_embedder():
    return get_embedder()


@st.cache_resource(show_spinner="Loading the search index…")
def load_index():
    return ClipIndex.load()


@st.cache_data
def load_audit() -> dict | None:
    path = config.AUDIT_DIR / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_robustness() -> dict | None:
    path = config.INDEX_DIR / "robustness.json"
    return json.loads(path.read_text()) if path.exists() else None


def data_uri(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(pipeline.thumbnail_jpeg(data, 160)).decode()


def badge(text: str, color: str) -> str:
    return f":{color}-badge[{text}]"


# ---------------------------------------------------------------- state

ss = st.session_state
ss.setdefault("files", [])  # list of {"name", "data"}
ss.setdefault("answers", {})  # name -> dict
ss.setdefault("batch", None)
ss.setdefault("zip", None)
ss.setdefault("pdf", None)


def add_files(items: list[tuple[str, bytes]], defaults: dict | None = None) -> list[str]:
    errors = []
    existing = {f["name"] for f in ss.files}
    for name, data in items:
        if name in existing:
            continue
        if len(ss.files) >= config.MAX_FILES:
            errors.append(f"Skipped {name}: the batch limit is {config.MAX_FILES} files.")
            continue
        try:  # validate early so the table only shows usable files
            pipeline.ingest(pipeline.Batch(), name, data)
        except pipeline.RejectedFile as exc:
            errors.append(str(exc))
            continue
        ss.files.append({"name": name, "data": data})
        ss.answers[name] = (defaults or {}).get(name, {"posted": True, "opted_out": False, "glazed": False,
                                                       "method": "human"})
    ss.batch = None
    return errors


# ---------------------------------------------------------------- header

st.title("🍌 " + config.APP_NAME)
st.markdown(f"**{config.TAGLINE}** Scan your images against a searchable image index, add Do Not Train metadata "
            "to copies, and get a dated record of what was found and done.")
st.caption("This app measures exposure and documents actions. It does not guarantee protection.")

with st.sidebar:
    st.header("About you")
    artist = st.text_input("Name (optional)", key="artist", help="Written into the tagged copies and the PDF log.")
    website = st.text_input("Website (optional)", key="website")
    st.divider()
    st.caption(f"{config.APP_NAME} {config.APP_VERSION}. Your images are processed in memory and are not stored.")
    if ss.files and st.button("Clear batch", width="stretch"):
        ss.files, ss.answers, ss.batch, ss.zip, ss.pdf = [], {}, None, None, None
        st.rerun()

# ---------------------------------------------------------------- 1. upload

st.subheader("1. Upload your work")
c1, c2 = st.columns([3, 1])
with c1:
    uploads = st.file_uploader(f"JPEG, PNG or WebP, up to {config.MAX_FILES} files", type=config.ALLOWED_TYPES,
                               accept_multiple_files=True, label_visibility="collapsed")
def load_demo() -> None:
    demo = [(f.name, f.read_bytes()) for f in sorted(config.DEMO_DIR.iterdir())
            if f.suffix.lower().lstrip(".") in config.ALLOWED_TYPES]
    add_files(demo, DEMO_ANSWERS)
    ss.artist = ss.get("artist") or "Demo Artist"


with c2:
    st.button("Load demo batch", width="stretch", help="Five sample pieces made for this demo",
              on_click=load_demo)
if uploads:
    for e in add_files([(u.name, u.getvalue()) for u in uploads]):
        st.warning(e)

if not ss.files:
    st.info("Upload images or click **Load demo batch** to try it.")
    st.stop()

# ---------------------------------------------------------------- 2. questions

st.subheader("2. Tell us about each piece")
st.caption("Self-reported answers are recorded as stated. The app can't verify opt-outs or style protection.")
rows = []
for f in ss.files:
    a = ss.answers[f["name"]]
    rows.append({
        "Preview": data_uri(f["data"]), "File": f["name"],
        "Publicly posted": a.get("posted", True), "Already opted out": a.get("opted_out", False),
        "Glazed / Nightshaded": a.get("glazed", False),
        "How it was made": METHOD_OPTIONS[a.get("method", "human")],
        "Changed AI output by hand": YN_OPTIONS[a.get("modified", "na")],
        "Your own arrangement": YN_OPTIONS[a.get("arranged", "na")],
    })
edited = st.data_editor(
    pd.DataFrame(rows), hide_index=True, width="stretch",
    key="editor-" + sha256_bytes("|".join(f["name"] for f in ss.files).encode())[:12],
    disabled=["Preview", "File"],
    column_config={
        "Preview": st.column_config.ImageColumn(width="small"),
        "How it was made": st.column_config.SelectboxColumn(options=list(METHOD_OPTIONS.values()), width="large",
                                                            required=True),
        "Changed AI output by hand": st.column_config.SelectboxColumn(options=list(YN_OPTIONS.values()),
                                                                      required=True),
        "Your own arrangement": st.column_config.SelectboxColumn(options=list(YN_OPTIONS.values()), required=True),
    },
)
inv_method = {v: k for k, v in METHOD_OPTIONS.items()}
inv_yn = {v: k for k, v in YN_OPTIONS.items()}
for _, r in edited.iterrows():
    ss.answers[r["File"]] = {
        "posted": bool(r["Publicly posted"]), "opted_out": bool(r["Already opted out"]),
        "glazed": bool(r["Glazed / Nightshaded"]), "method": inv_method[r["How it was made"]],
        "modified": inv_yn[r["Changed AI output by hand"]], "arranged": inv_yn[r["Your own arrangement"]],
    }

# ---------------------------------------------------------------- 3. run

st.subheader("3. Scan and protect")
if st.button("Run scan", type="primary"):
    batch = pipeline.Batch(artist=artist.strip(), website=website.strip())
    with st.status("Working…", expanded=True) as status:
        for f in ss.files:
            p = pipeline.ingest(batch, f["name"], f["data"])
            a = ss.answers[f["name"]]
            p.publicly_posted, p.opted_out, p.style_protected = a["posted"], a["opted_out"], a["glazed"]
            p.ai_answers = {k: a[k] for k in ("method", "modified", "arranged")}
        st.write(f"Hashed {len(batch.pieces)} file(s) (SHA-256).")
        embedder, index = load_embedder(), load_index()
        batch.log.add("start", "done_by_app", f"{config.APP_NAME} {config.APP_VERSION}: processing "
                      f"{len(batch.pieces)} piece(s) for {batch.artist or 'an unnamed artist'}.")
        pipeline.scan(batch, index, embedder)
        st.write(f"Compared each image with {len(index):,} indexed images using CLIP.")
        pipeline.tag(batch)
        st.write("Added Do Not Train metadata to copies and read it back.")
        pipeline.assess_rights(batch)
        pipeline.score(batch)
        pipeline.verify_originals(batch)
        st.write("Scored exposure and checked that originals are unchanged.")
        ss.zip = build_zip(batch)
        ss.pdf = build_pdf(batch, load_audit())
        batch.log.add("export", "done_by_app", "Built the ZIP of tagged copies and the PDF log.")
        ss.batch = batch
        status.update(label="Done", state="complete", expanded=False)

batch: pipeline.Batch | None = ss.batch
if batch is None:
    st.info("Click **Run scan** to scan, tag and score your batch.")
    st.stop()

current = [(f["name"], ss.answers[f["name"]]) for f in ss.files]
ran = [(p.filename, p) for p in batch.pieces]
if len(current) != len(ran) or any(
        a["posted"] != p.publicly_posted or a["opted_out"] != p.opted_out or a["glazed"] != p.style_protected
        or {k: a[k] for k in ("method", "modified", "arranged")} != p.ai_answers
        for (_, a), (_, p) in zip(current, ran)):
    st.warning("You changed the batch or your answers after the last run. Click **Run scan** again to update.")

# ---------------------------------------------------------------- results

tab_report, tab_audit, tab_cov, tab_how, tab_dl = st.tabs(
    ["📋 Report", "🎨 Style audit", "🗺️ Coverage map", "🔍 How it works", "⬇️ Download"])

with tab_report:
    labels = [p.label for p in batch.pieces]
    matches = [p.match.label for p in batch.pieces]
    m = st.columns(6)
    m[0].metric("Pieces", len(batch.pieces))
    m[1].metric("High exposure", labels.count("High"))
    m[2].metric("Medium", labels.count("Medium"))
    m[3].metric("Lower", labels.count("Lower"))
    m[4].metric("Strong matches", matches.count("strong"))
    m[5].metric("Tags verified", f"{sum(p.tag_verified for p in batch.pieces)}/{len(batch.pieces)}")

    info = batch.index_info
    st.caption(f"Searched: {info['name']} ({info['size']:,} images: "
               + ", ".join(f"{s['name']} {s['count']:,}" for s in info["sources"])
               + "). Not searched: everything else, including real AI training datasets.")

    table = pd.DataFrame([{
        "Preview": data_uri(p.original_bytes), "File": p.filename,
        "Dataset scan": MATCH_TEXT[p.match.label], "Top similarity": p.match.top_score,
        "Exposure score": p.score, "Level": p.label,
        "Needs you": sum(s.status == "needs_you" for s in p.next_steps),
    } for p in batch.pieces]).sort_values("Exposure score", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Preview": st.column_config.ImageColumn(width="small"),
        "Top similarity": st.column_config.NumberColumn(format="%.3f"),
        "Exposure score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
    })

    for p in sorted(batch.pieces, key=lambda x: -x.score):
        header = f"**{p.filename}**: {p.score}/100 · {p.label} · {MATCH_TEXT[p.match.label]}"
        with st.expander(header):
            st.markdown(badge(f"{p.label} exposure", LABEL_BADGE[p.label]) + " "
                        + badge(MATCH_TEXT[p.match.label], {"strong": "red", "possible": "orange",
                                                             "none": "green"}[p.match.label]))
            left, right = st.columns([1, 2])
            with left:
                st.image(pipeline.thumbnail_jpeg(p.original_bytes, 400), caption="Your piece")
                st.markdown("**Score breakdown**")
                for k, v in p.breakdown.items():
                    st.markdown(f"- {config.SIGNAL_LABELS[k]}: **+{v}**")
            with right:
                st.markdown("**Nearest images in the index**")
                ncols = st.columns(len(p.match.neighbors))
                for col, n in zip(ncols, p.match.neighbors):
                    thumb = config.INDEX_DIR / n["thumb"]
                    if thumb.exists():
                        col.image(str(thumb))
                    col.caption(f"{n['score']:.3f}  \n{(n['title'] or '')[:40]}")
                if p.match.upgraded_by_phash:
                    st.caption(f"Upgraded to strong: perceptual hash distance {p.match.phash_distance}.")
                st.markdown("**Next steps**")
                for s in p.next_steps:
                    line = f"{badge(STATUS_TEXT[s.status], STATUS_BADGE[s.status])} **{s.title}.** {s.detail}"
                    if s.link:
                        line += f" [Open ↗]({s.link})"
                    st.markdown(line)
            st.markdown("**Rights summary**")
            r = rights.summarize(p.ai_answers)
            st.markdown(f"{badge(r['status'], 'blue')} {r['summary']}")
            st.caption(r["disclaimer"] + " Sources: " + "; ".join(f"[{s['title']}]({s['url']})" for s in r["sources"]))
            st.markdown("**Record**")
            st.code(f"original     {p.sha256_original}\ntagged copy  {p.sha256_tagged or 'n/a'}\n"
                    f"tag engine   {p.tag_engine}  verified={p.tag_verified}", language=None)

    with st.expander("Limits of this report"):
        for line in config.LIMITATIONS:
            st.markdown(f"- {line}")

with tab_audit:
    audit = load_audit()
    if not audit:
        st.info("No audit results yet. Run `python audit/generate.py` and `python audit/analyze.py`.")
    else:
        st.markdown(f"#### Does naming **{audit['artist']}** pull a model's output toward their work?")
        st.markdown(
            "We asked each model for the same subjects three ways: *in the style of the artist*, as a generic "
            "*painting*, and as the artist's genre (*Japanese woodblock print*) without the name. **Style lift** "
            "is how much closer the styled images are to the artist's reference works (CLIP similarity). "
            "Bars show 95% bootstrap intervals.")
        st.caption("Demo uses a public-domain historical artist as a known case. A real user would audit their "
                   "own name only.")
        recs = []
        for mdl in audit["models"]:
            recs.append({"Model": mdl["model"], "Control": "vs. generic painting", "Lift": mdl["lift"],
                         "lo": mdl["ci_low"], "hi": mdl["ci_high"]})
            recs.append({"Model": mdl["model"], "Control": "vs. same genre, no name", "Lift": mdl["genre_lift"],
                         "lo": mdl["genre_ci_low"], "hi": mdl["genre_ci_high"]})
        df = pd.DataFrame(recs)
        base = alt.Chart(df).encode(y=alt.Y("Model:N", title=None), yOffset="Control:N",
                                    color=alt.Color("Control:N", legend=alt.Legend(orient="bottom", title=None)))
        chart = (base.mark_bar().encode(x=alt.X("Lift:Q", title="Style lift (CLIP similarity)"),
                                        tooltip=["Model", "Control", alt.Tooltip("Lift:Q", format="+.3f")])
                 + base.mark_errorbar().encode(x="lo:Q", x2="hi:Q"))
        st.altair_chart(chart.properties(height=60 + 70 * len(audit["models"])), width="stretch")
        cols = st.columns(len(audit["models"]))
        for col, mdl in zip(cols, audit["models"]):
            col.metric(mdl["model"], f"{mdl['lift']:+.3f}", help="Lift vs. generic painting")
            col.caption(f"{mdl['n_styled']} styled images · near-copy rate {mdl['near_copy_rate']:.0%} "
                        f"(max similarity to one reference work {mdl['max_similarity_to_a_reference']:.3f})")
        st.markdown(f"**Sample outputs:** *{audit['grid']['subject']}*")
        names = {"styled": "In the style of", "generic": "Generic painting", "genre": "Genre, no name"}
        for row in audit["grid"]["rows"]:
            st.caption(row["model"])
            gcols = st.columns(4)
            for gcol, t in zip(gcols, ["styled", "generic", "genre"]):
                if t in row:
                    gcol.image(str(config.AUDIT_DIR / row[t]), caption=names[t])
        if audit.get("reference_thumbs"):
            st.caption(f"Reference works ({audit['reference_count']} from {audit['reference_source']})")
            rcols = st.columns(len(audit["reference_thumbs"]))
            for rcol, t in zip(rcols, audit["reference_thumbs"]):
                rcol.image(str(config.AUDIT_DIR / t))
        st.info(audit["caption"])

with tab_cov:
    cov = coverage.load_coverage()
    st.markdown("#### Which protections apply to which AI developers?")
    st.caption(f"As of {cov['as_of']}. **Unknown** means we found no public statement. "
               "Yes for robots.txt covers future crawls only; it doesn't remove data already collected.")
    ct = coverage.as_table()
    palette = {"yes": "background-color:#D1FADF", "no": "background-color:#FEE4E2", "unknown": "color:#8A8577"}
    st.dataframe(ct.style.map(lambda v: palette.get(v, "")), width="stretch")
    counts = coverage.counts()
    st.markdown(f"**{counts['unknown']} of {sum(counts.values())}** cells are unknown. Consent signals only work "
                "if AI developers choose to honor them.")
    with st.expander("Sources"):
        for s in coverage.sources():
            st.markdown(f"- **{s['protection']} / {s['trainer']}**: {s['value']}. {s['note']} "
                        f"[source]({s['source_url']})" + ("" if s["verified"] else " *(unverified)*"))

with tab_how:
    st.markdown("#### Exposure score")
    st.markdown("Points are added for each signal. The weights are a design choice:")
    st.table(pd.DataFrame([{"Signal": config.SIGNAL_LABELS[k], "Points": v} for k, v in config.WEIGHTS.items()]))
    st.caption(f"High ≥ {config.LABEL_HIGH} · Medium {config.LABEL_MEDIUM}–{config.LABEL_HIGH - 1} · "
               f"Lower < {config.LABEL_MEDIUM}. Strong and possible matches never both apply, so the maximum is 100. "
               "The score describes the copy already posted online, which has no tag.")
    st.markdown("#### Dataset scan")
    st.markdown(f"Each image is embedded with CLIP **{config.CLIP_MODEL}** and compared to every image in the index "
                f"(cosine similarity). **Strong** ≥ {config.T_STRONG:.2f}, **possible** ≥ {config.T_POSSIBLE:.2f}. "
                f"A possible match whose perceptual hash is within {config.PHASH_MAX_DISTANCE} bits of the "
                "neighbor is upgraded to strong. Thresholds were measured, not guessed: see `index/calibrate.py`.")
    rob = load_robustness()
    if rob:
        st.markdown(f"**Robustness:** similarity of *{rob['piece']}* to edited copies of itself")
        rdf = pd.DataFrame(rob["rows"])
        rdf["kind"] = ["unrelated" if "unrelated" in v else "copy" for v in rdf["variant"]]
        bars = alt.Chart(rdf).mark_bar().encode(
            x=alt.X("similarity:Q", scale=alt.Scale(domain=[0, 1]), title="CLIP cosine similarity"),
            y=alt.Y("variant:N", sort=None, title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["copy", "unrelated"], range=["#C99A06", "#8A8577"]),
                            legend=None),
            tooltip=["variant", alt.Tooltip("similarity:Q", format=".3f")])
        rules = alt.Chart(pd.DataFrame({"t": [rob["t_strong"], rob["t_possible"]],
                                        "name": ["strong", "possible"]})).mark_rule(strokeDash=[4, 3]).encode(
            x="t:Q", tooltip=["name", "t"])
        st.altair_chart((bars + rules).properties(height=28 * len(rdf)), width="stretch")
        st.caption("Dashed lines: possible and strong thresholds. An exact file hash would only match the original.")
    st.markdown("#### Do Not Train tag")
    st.markdown(f"Copies get the IPTC/PLUS **DataMining** property set to `{config.DMI_PROHIBITED_AIML}`, plus a "
                f"plain-text notice in `plus:OtherConstraints`, `xmpRights:UsageTerms` and `dc:rights`. Pixels "
                "are untouched. Each tag is read back to verify it.")
    st.markdown("#### Limits")
    for line in config.LIMITATIONS:
        st.markdown(f"- {line}")

with tab_dl:
    st.markdown("#### Your files")
    d1, d2 = st.columns(2)
    stamp = batch.created_at[:10]
    d1.download_button("⬇️ Tagged copies (ZIP)", ss.zip, file_name=f"artguard-tagged-{stamp}.zip",
                       mime="application/zip", width="stretch", type="primary")
    d2.download_button("⬇️ PDF log", ss.pdf, file_name=f"artguard-log-{stamp}.pdf", mime="application/pdf",
                       width="stretch")
    st.markdown("Keep this fingerprint of your PDF somewhere separate (an email to yourself works). It lets you "
                "show later that the file hasn't changed.")
    st.code(f"PDF SHA-256  {sha256_bytes(ss.pdf)}", language=None)
    st.caption("The ZIP holds tagged copies only; your originals were not modified. Check a tag with: "
               "`exiftool -XMP-plus:DataMining <file>`")
    with st.expander("Action log"):
        names = {p.id: p.filename for p in batch.pieces}
        st.dataframe(pd.DataFrame([{"Time (UTC)": a.ts_utc, "Piece": names.get(a.piece_id, "batch"),
                                    "Step": a.step, "Status": STATUS_TEXT[a.status], "Detail": a.detail}
                                   for a in batch.log.actions]), hide_index=True, width="stretch")
    st.warning("Limits: " + " ".join(config.LIMITATIONS[:4]))
