## Inspiration

Independent artists can't easily tell whether their work was collected for AI training, and protections are spread across separate tools that only work if AI developers honor them. We wanted one place that measures exposure honestly and leaves a record.

## What it does

ArtGuard Ledger takes a batch of images and:

- **Scans** each one against a searchable image index with CLIP and labels it strong, possible or no match.
- **Tags** copies with the IPTC "Do Not Train" data-mining property, reads every tag back, and leaves originals untouched.
- **Scores** exposure from 0 to 100 with published weights, and ranks next steps as *Done by app*, *Needs you* or *Can't verify*.
- **Explains** likely copyright status in plain English from US Copyright Office guidance (not legal advice).
- **Audits** whether image models drift toward an artist's style when prompted with their name.
- **Exports** a ZIP of tagged copies and a dated PDF log with SHA-256 hashes of every original and copy.

## How we built it

Python, Streamlit and OpenCLIP (ViT-B-32). The index holds 2,914 public-domain artworks from the Art Institute of Chicago plus simulated "scraped" copies of our demo pieces. We **measured** thresholds instead of guessing them: edited copies against 295 held-out images that were never indexed. Strong ≥ 0.96 and possible ≥ 0.90 flag only 1% of unrelated images. A perceptual hash upgrades a possible match to strong when the pixels clearly agree.

Tagging uses ExifTool, or our own pure-Python XMP writer for JPEG, PNG and WebP that never re-encodes pixels. Tests confirm identical pixels and ExifTool-readable tags.

For the style audit, we generated 192 images with Stable Diffusion 1.5 and SD-Turbo (8 subjects × 4 seeds). Each subject was prompted "in the style of Katsushika Hokusai", as a generic painting, and as a Japanese woodblock print without the name. **Style lift** is the CLIP similarity to Hokusai's reference works, styled minus control, with bootstrap confidence intervals. SD 1.5 moves sharply toward Hokusai when named (+0.33 vs. a generic painting), but the name adds almost nothing beyond the genre. With SD-Turbo, the name pulls *less* than the genre alone (−0.12, 95% CI −0.15 to −0.08). The models differ in many ways, so this is a signal, not proof.

## Challenges

Museum artworks look alike to CLIP: unrelated pairs reach 0.95 similarity, so off-the-shelf thresholds produced false matches. We also learned that the honest default for "does this AI developer honor this opt-out?" is *unknown*, and most of our coverage map is exactly that.

## What we learned

A scan covers only what it searches, one upload can strip metadata, and an opt-out is a request, not a lock. We show these limits on every screen and in the PDF.

## What's next

Search real dataset indexes, add C2PA content credentials, add independent timestamps for the log, and let artists run the style audit on their own names.

*Built with Claude Code (Anthropic). Image models: Stable Diffusion 1.5 and SD-Turbo.*
