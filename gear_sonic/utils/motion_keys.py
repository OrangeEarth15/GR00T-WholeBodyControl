"""Read an ordered, exact motion-key list for evaluation."""

import json
from pathlib import Path


def load_motion_keys(path: str | Path) -> list[str]:
    """Read a nonempty JSON list of unique motion names, preserving its order."""
    keys = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(keys, list) or not keys:
        raise ValueError("Evaluation motion keys must be a nonempty JSON list")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("Each evaluation motion key must be a nonempty string")
    if len(set(keys)) != len(keys):
        raise ValueError("Evaluation motion keys must not contain duplicates")
    return keys
