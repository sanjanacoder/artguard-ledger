"""Coverage map: which protections apply to which AI trainers (yes / no / unknown)."""
import json
from functools import lru_cache

import pandas as pd

import config

UNKNOWN = {"value": "unknown", "note": "No public statement found.", "source_url": None, "verified": False}


@lru_cache(maxsize=1)
def load_coverage() -> dict:
    return json.loads((config.DATA_DIR / "coverage.json").read_text())


def cell(protection_key: str, trainer: str) -> dict:
    return load_coverage()["cells"].get(protection_key, {}).get(trainer, UNKNOWN)


def as_table() -> pd.DataFrame:
    """Rows = protections, columns = trainers, values = yes/no/unknown."""
    cov = load_coverage()
    rows = {}
    for p in cov["protections"]:
        rows[p["name"]] = {t: cell(p["key"], t)["value"] for t in cov["trainers"]}
    return pd.DataFrame.from_dict(rows, orient="index")


def sources() -> list[dict]:
    cov = load_coverage()
    out = []
    for p in cov["protections"]:
        for t in cov["trainers"]:
            c = cell(p["key"], t)
            if c["value"] != "unknown":
                out.append({"protection": p["name"], "trainer": t, **c})
    return out


def counts() -> dict[str, int]:
    values = as_table().to_numpy().ravel().tolist()
    return {v: values.count(v) for v in ("yes", "no", "unknown")}
