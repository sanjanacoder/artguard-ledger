import json
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def load_resources() -> dict[str, dict]:
    data = json.loads((config.DATA_DIR / "resources.json").read_text())
    return {r["key"]: r for r in data["resources"]}


def resource(key: str) -> dict:
    return load_resources()[key]
