

from __future__ import annotations

import logging
import math
import time

import config
import embedding
import jsonio
import storage
from generation.mid_long_evidence import (
    TYPED_LONG_TABLES,
    group_typed_longs_by_mid,
    mid_embedding_retrieval_text,
    text_sha256,
)

logger = logging.getLogger("enriched_embeddings")

_EMBED_MAX_ATTEMPTS = 3


def _valid_vector(vector: object) -> bool:
    return (isinstance(vector, list) and bool(vector)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in vector)
            and any(value != 0 for value in vector))


def build_enriched_mid_embeddings(
    *,
    conversation_id: str | None = None,
    batch_size: int = 50,
    restart: bool = False,
    prune_orphans: bool = False,
    output_path: str | None = None,
    strict: bool = False,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prune_orphans and conversation_id is not None:
        raise ValueError("prune_orphans requires a full-store run")

    output_path = output_path or config.MID_ENRICHED_EMBEDDINGS_PATH
    mids = [
        record
        for record in storage.all_mid_memories()
        if conversation_id is None or record.get("conversation_id") == conversation_id
    ]
    longs_by_mid = group_typed_longs_by_mid(
        {table: storage.all_typed_longs(table) for table in TYPED_LONG_TABLES}
    )
    ids = [str(record.get("id") or "") for record in mids]
    if any(not record_id for record_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("mid embedding input contains missing or duplicate ids")

    existing = jsonio.read_json(output_path, default=[]) or []
    if not isinstance(existing, list):
        raise ValueError(f"expected a JSON array at {output_path}")
    by_id = {
        str(record.get("id")): record
        for record in existing
        if str(record.get("id") or "")
    }

    pruned = 0
    if prune_orphans:
        active = set(ids)
        pruned = sum(record_id not in active for record_id in by_id)
        by_id = {k: v for k, v in by_id.items() if k in active}

    pending: list[tuple[str, str, str, int]] = []
    cached = 0
    for mid in mids:
        record_id = str(mid["id"])
        children = longs_by_mid.get(record_id, [])
        text = mid_embedding_retrieval_text(mid, children)
        digest = text_sha256(text)
        previous = by_id.get(record_id) or {}
        reusable = (
            not restart
            and previous.get("embedding_model") == config.EMBEDDING_MODEL
            and previous.get("text_sha256") == digest
            and isinstance(previous.get("embedding"), list)
            and bool(previous.get("embedding"))
            and (not strict or _valid_vector(previous.get("embedding")))
        )
        if reusable:
            cached += 1
        else:
            pending.append((record_id, text, digest, len(children)))

    logger.info(
        "selected %d mid(s): %d reusable, %d to embed", len(mids), cached, len(pending)
    )

    embedded = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = None
        valid_batch = False
        for attempt in range(1, _EMBED_MAX_ATTEMPTS + 1):
            vectors = embedding.embed_texts([text for _, text, _, _ in batch])
            valid_batch = bool(vectors) and len(vectors) == len(batch)
            if strict:
                valid_batch = (isinstance(vectors, list) and valid_batch
                               and all(_valid_vector(vector) for vector in vectors)
                               and len({len(vector) for vector in vectors}) == 1)
            if valid_batch:
                break
            if attempt < _EMBED_MAX_ATTEMPTS:
                logger.warning(
                    "embedding chunk failed (%d/%d); retrying",
                    attempt,
                    _EMBED_MAX_ATTEMPTS,
                )
                time.sleep(attempt)
        if not valid_batch:
            raise RuntimeError(
                f"failed to embed chunk starting at {start} after "
                f"{_EMBED_MAX_ATTEMPTS} attempts"
            )
        for (record_id, _text, digest, child_count), vector in zip(batch, vectors):
            mid = next(record for record in mids if str(record["id"]) == record_id)
            by_id[record_id] = {
                "id": record_id,
                "conversation_id": mid.get("conversation_id"),
                "embedding_model": config.EMBEDDING_MODEL,
                "text_sha256": digest,
                "child_long_count": child_count,
                "embedding": vector,
            }
        embedded += len(batch)
        # Persist after every chunk so an interrupted run keeps the vectors it paid for.
        jsonio.atomic_write_json(output_path, list(by_id.values()))
        logger.info("embedded %d/%d pending mid(s)", embedded, len(pending))

    if not embedded:
        jsonio.atomic_write_json(output_path, list(by_id.values()))

    return {
        "selected": len(mids),
        "cached": cached,
        "embedded": embedded,
        "pruned": pruned,
        "sidecar_rows": len(by_id),
        "output_path": output_path,
    }
