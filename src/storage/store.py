

from __future__ import annotations

from pathlib import Path

import config
import jsonio

_TYPED_TABLES = {"long_core", "long_episodic", "long_knowledge"}


def _read(table: str) -> list[dict]:
    rows = jsonio.read_json(str(Path(config.DATA_DIR) / f"{table}.json"), default=[])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Memory table must contain an array of records: {table}")
    return rows


def all_mid_memories() -> list[dict]:
    return _read("mid_memories")


def all_mid_relations() -> list[dict]:
    return _read("mid_relations")


def all_typed_longs(table: str) -> list[dict]:
    if table not in _TYPED_TABLES:
        raise ValueError(f"Unknown typed Long table: {table}")
    return _read(table)
