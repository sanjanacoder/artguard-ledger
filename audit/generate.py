"""Generate audit images (offline; needs a GPU or Apple Silicon).

    pip install -r requirements-audit.txt
    python audit/generate.py                # all models in prompts.yaml
    python audit/generate.py --model stabilityai/sd-turbo

Writes audit/outputs/<model-slug>/<template>_<subject>_<seed>.png and a
manifest.json next to them. Existing images are skipped, so it can resume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

OUT = config.AUDIT_DIR / "outputs"


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_setup() -> dict:
    return yaml.safe_load((config.AUDIT_DIR / "prompts.yaml").read_text())


def prompts(setup: dict):
    for t_key, template in setup["templates"].items():
        for s_i, subject in enumerate(setup["subjects"]):
            yield t_key, s_i, template.format(subject=subject, artist=setup["artist"])


def device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def generate_model(setup: dict, model: dict) -> None:
    import torch
    from diffusers import AutoPipelineForText2Image

    device, dtype = device_and_dtype()
    pipe = AutoPipelineForText2Image.from_pretrained(model["id"], torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    folder = OUT / slug(model["id"])
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    jobs = [(t, s, p, seed) for t, s, p in prompts(setup) for seed in setup["seeds"]]
    for n, (t_key, s_i, prompt, seed) in enumerate(jobs, 1):
        name = f"{t_key}_{s_i}_{seed}.png"
        if (folder / name).exists():
            continue
        gen = torch.Generator("cpu").manual_seed(seed)
        result = pipe(prompt, num_inference_steps=model["steps"], guidance_scale=model["guidance"],
                      generator=gen, height=512, width=512)
        result.images[0].save(folder / name)
        nsfw = getattr(result, "nsfw_content_detected", None)
        manifest[name] = {"template": t_key, "subject": s_i, "seed": seed, "prompt": prompt,
                          "flagged": bool(nsfw[0]) if nsfw else False}
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{model['label']}] {n}/{len(jobs)} {name}", flush=True)
    meta = {"model_id": model["id"], "label": model["label"], "steps": model["steps"],
            "guidance": model["guidance"], "device": device, "dtype": str(dtype), "size": 512}
    (folder / "model.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="only this model id")
    args = ap.parse_args()
    setup = load_setup()
    for model in setup["models"]:
        if args.model and model["id"] != args.model:
            continue
        generate_model(setup, model)


if __name__ == "__main__":
    main()
