"""Plain-English copyright summaries from a fixed rule table (no LLM)."""
from __future__ import annotations

from functools import lru_cache

import yaml

import config


@lru_cache(maxsize=1)
def load_rules() -> dict:
    # BaseLoader keeps yes/no as strings instead of booleans.
    return yaml.load((config.DATA_DIR / "rights_rules.yaml").read_text(), Loader=yaml.BaseLoader)


def questions() -> dict:
    return load_rules()["questions"]


def sources() -> list[dict]:
    return load_rules()["sources"]


def normalize(answers: dict | None) -> dict:
    answers = answers or {}
    method = answers.get("method") or "human"
    out = {"method": method, "modified": answers.get("modified") or "na", "arranged": answers.get("arranged") or "na"}
    if method in ("human", "ai_minor"):
        out["modified"] = out["arranged"] = "na"
    return out


def summarize(answers: dict | None) -> dict:
    """Return {key, status, summary, disclaimer, sources} for the first matching rule."""
    answers = normalize(answers)
    for rule in load_rules()["rules"]:
        if all(v == "any" or answers.get(k) == v for k, v in rule["when"].items()):
            return {
                "key": rule["key"],
                "status": rule["status"],
                "summary": " ".join(rule["summary"].split()),
                "disclaimer": config.NOT_LEGAL_ADVICE,
                "sources": sources(),
            }
    raise LookupError(f"No rights rule matches {answers}")
