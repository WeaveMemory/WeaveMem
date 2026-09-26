
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def record_path(run_dir: Path, stage: str, record_id: str) -> Path:
    return run_dir / stage / (hashlib.sha256(record_id.encode()).hexdigest() + ".json")


def load_record(run_dir: Path, stage: str, record_id: str, input_digest: str):
    path = record_path(run_dir, stage, record_id)
    if not path.exists():
        return None
    result = checked_record(path)
    if result.get("id") != record_id or result.get("input_sha256") != input_digest:
        raise ValueError(f"Stale {stage} artifact for {record_id}; use a new output directory")
    return result


def save_record(run_dir: Path, stage: str, record_id: str, input_digest: str,
                result: dict) -> dict:
    record = {**result, "id": record_id, "input_sha256": input_digest}
    record["record_sha256"] = digest(record)
    write_json(record_path(run_dir, stage, record_id), record)
    return record


def checked_record(path: Path) -> dict:
    record = read_json(path)
    if not isinstance(record, dict):
        raise ValueError("Invalid stage record")
    expected = record.get("record_sha256")
    actual = digest({key: value for key, value in record.items() if key != "record_sha256"})
    if expected != actual:
        raise ValueError(f"Stage artifact changed or is incomplete: {path.name}")
    return record


def implementation_digest(root: Path) -> str:
    files = {}
    for directory in (root / "src", root / "benchmarks"):
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in {".py", ".txt"}
                    and not any(part.startswith(".") or part == "__pycache__"
                                for part in relative.parts)):
                files[relative.as_posix()] = file_digest(path)
    return digest(files)
