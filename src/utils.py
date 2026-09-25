#src/utils.py = paths + JSON/JSONL reading/writing + JSON-safe serialization
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from numbers import Integral, Number
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_safe_filename_label(label: str) -> str:
    return (
        str(label)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, bool):
        return value

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, Number):
        if isinstance(value, Integral):
            return int(value)

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(v)
            for v in value
        ]

    return value


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            make_json_serializable(obj),
            f,
            ensure_ascii=False,
            indent=2,
        )


def read_json(path: str | Path) -> Any:
    """
    Read a JSON file.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            safe_record = make_json_serializable(record)
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    records: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    safe_record = make_json_serializable(record)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def count_jsonl_records(path: str | Path) -> int:
    path = Path(path)

    if not path.exists():
        return 0

    n = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1

    return n