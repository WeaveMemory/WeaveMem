

from __future__ import annotations

import os
import threading

import config
import jsonio
import storage

from .graph_navigation import clear_enriched_embedding_cache, search_graph_paths
from .result import RetrievalResult

_TYPED_TABLES = ("long_core", "long_episodic", "long_knowledge")
_CACHE_LOCK = threading.RLock()
_SNAPSHOT_SIGNATURE: tuple | None = None
_SNAPSHOT: dict[str, list[dict]] | None = None
_SCOPED_CACHE: dict[tuple, dict] = {}


def _table_path(table: str) -> str:
    override = {
        "mid_memories": config.MID_MEMORIES_OVERRIDE_PATH,
        "mid_relations": config.MID_RELATIONS_OVERRIDE_PATH,
    }.get(table)
    return override or os.path.join(config.DATA_DIR, f"{table}.json")


def _load_mid_table(table: str) -> list[dict]:
    override = {
        "mid_memories": config.MID_MEMORIES_OVERRIDE_PATH,
        "mid_relations": config.MID_RELATIONS_OVERRIDE_PATH,
    }[table]
    if not override:
        loader = storage.all_mid_memories if table == "mid_memories" else storage.all_mid_relations
        return loader()
    rows = jsonio.read_json(override, default=None)
    if not isinstance(rows, list):
        raise ValueError(f"The {table} override must contain a JSON array")
    return [row for row in rows if isinstance(row, dict)]


def _storage_signature() -> tuple:
    """Invalidate cached tables when a file is replaced or updated."""
    values = []
    for table in ("mid_memories", "mid_relations", *_TYPED_TABLES):
        path = _table_path(table)
        try:
            stat = os.stat(path)
            values.append((path, stat.st_size, stat.st_mtime_ns))
        except OSError:
            values.append((path, None, None))
    return tuple(values)


def clear_retrieval_cache() -> None:
    """Drop snapshots, scoped views, and the enriched Mid embedding sidecar cache."""
    global _SNAPSHOT_SIGNATURE, _SNAPSHOT
    with _CACHE_LOCK:
        _SNAPSHOT_SIGNATURE = None
        _SNAPSHOT = None
        _SCOPED_CACHE.clear()
    clear_enriched_embedding_cache()


def _load_snapshot(signature: tuple) -> dict[str, list[dict]]:
    global _SNAPSHOT_SIGNATURE, _SNAPSHOT
    with _CACHE_LOCK:
        if _SNAPSHOT is None or _SNAPSHOT_SIGNATURE != signature:
            _SNAPSHOT = {
                "mids": _load_mid_table("mid_memories"),
                "mid_rels": _load_mid_table("mid_relations"),
                **{table: storage.all_typed_longs(table) for table in _TYPED_TABLES},
            }
            _SNAPSHOT_SIGNATURE = signature
            _SCOPED_CACHE.clear()
        return _SNAPSHOT


def _load_scoped(conversation_id: str | None, user_id: str | None) -> dict:
    """Keep the selected Mid nodes, their graph edges, and typed child evidence."""
    signature = _storage_signature()
    cache_key = (signature, conversation_id, user_id)
    with _CACHE_LOCK:
        cached = _SCOPED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    snapshot = _load_snapshot(signature)
    mids = [
        record
        for record in snapshot["mids"]
        if (conversation_id is None or record.get("conversation_id") == conversation_id)
        and (user_id is None or record.get("user_id") == user_id)
    ]
    mid_ids = {record["id"] for record in mids}

    def typed_record_in_scope(record: dict) -> bool:
        # Session-level typed records may omit mid_id. The enrichment helper
        # associates these records to their owning Mid from session provenance.
        direct_conversation = str(record.get("conversation_id") or "").strip()
        if direct_conversation:
            return (conversation_id is None or direct_conversation == conversation_id) and (
                user_id is None or record.get("user_id") == user_id
            )
        return record.get("mid_id") in mid_ids

    scoped = {
        "mids": mids,
        "mid_by_id": {record["id"]: record for record in mids},
        "mid_rels": [
            relation
            for relation in snapshot["mid_rels"]
            if relation.get("source_id") in mid_ids and relation.get("target_id") in mid_ids
        ],
        **{
            table: [record for record in snapshot[table] if typed_record_in_scope(record)]
            for table in _TYPED_TABLES
        },
    }
    with _CACHE_LOCK:
        return _SCOPED_CACHE.setdefault(cache_key, scoped)


def search_memory_result(
    question: str,
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    """Retrieve answer memories and a serializable trace without mutating storage."""
    return search_graph_paths(question, _load_scoped(conversation_id, user_id), top_k=top_k)


def search_memory(
    question: str,
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Return only answer memories from the same formal retrieval pipeline."""
    return search_memory_result(
        question, conversation_id=conversation_id, user_id=user_id, top_k=top_k
    ).memories
