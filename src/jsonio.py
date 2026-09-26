

from __future__ import annotations

import json
import os
from typing import Any


def atomic_write_json(path: str, obj: Any) -> None:
    """Write obj as JSON to path atomically (temp file + os.replace)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON from path; return default if the file does not exist."""
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
