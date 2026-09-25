from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

#src/config.py = YAML config loading/saving/CLI overrides only

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = deepcopy(config)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        value = _parse_value(raw_value)

        parts = key.split(".")
        cur = result
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value

    return result

def parse_optional_int(value: Any) -> int | None:
    """
    Parse an optional integer config value.

    Useful for YAML values such as:
        top_k: null
        top_k: 50
    """
    if value is None:
        return None

    return int(value)


def parse_optional_float(value: Any) -> float | None:
    """
    Parse an optional float config value.

    Useful for YAML values such as:
        top_p: null
        top_p: 0.95
    """
    if value is None:
        return None

    return float(value)