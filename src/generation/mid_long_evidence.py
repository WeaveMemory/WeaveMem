
from __future__ import annotations

from collections import defaultdict
import hashlib
import json


TYPED_LONG_TABLES = (
    "long_core",
    "long_episodic",
    "long_knowledge",
)


def normalize_text(value) -> str:
    if isinstance(value, list):
        return "; ".join(item for item in (normalize_text(item) for item in value) if item)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(str(value or "").split())


def group_typed_longs_by_mid(tables: dict[str, list[dict]]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for table in TYPED_LONG_TABLES:
        for record in tables.get(table) or []:
            record_id = normalize_text(record.get("id"))
            mid_id = normalize_text(record.get("mid_id"))
            if not record_id or not mid_id:
                continue
            if record_id in seen:
                raise ValueError(f"duplicate typed long-memory id: {record_id}")
            seen.add(record_id)
            grouped[mid_id].append(record)
    return dict(grouped)


def _append_typed_child_lines(
    lines: list[str],
    child_longs: list[dict],
    *,
    projection: str,
) -> None:
    """Append child-long fields for one Mid retrieval projection.

    Core memories deliberately keep the historical ``content + tags`` projection.
    Episodic and knowledge memories have distinct lexical and dense projections so
    exact-match fields need not enter the embedding input.
    """
    if projection not in {"bm25", "embedding", "rerank"}:
        raise ValueError(f"unknown Mid retrieval projection: {projection}")
    for record in child_longs:
        memory_type = normalize_text(record.get("type")) or "unknown"
        content = normalize_text(record.get("content"))
        tags = normalize_text(record.get("tags"))
        parts = [f"[{memory_type}]"]

        if projection == "rerank":
            if not content and not tags:
                continue
            parts.append(content)
            if tags:
                parts.append(f"Tags: {tags}")
            if memory_type == "episodic":
                original_time_expression = normalize_text(record.get("originalTimeExpression"))
                time_precision = normalize_text(record.get("timePrecision"))
                event_time = normalize_text(record.get("eventTime"))
                mention_time = normalize_text(record.get("mentionTime"))
                if original_time_expression:
                    parts.append(f"Original time expression: {original_time_expression}")
                if time_precision:
                    parts.append(f"Time precision: {time_precision}")
                if event_time:
                    parts.append(f"Event time: {event_time}")
                if mention_time:
                    parts.append(f"Mention time: {mention_time}")
        elif memory_type == "episodic":
            context = normalize_text(record.get("context"))
            event_type_l1 = normalize_text(record.get("eventTypeL1"))
            event_type_l2 = normalize_text(record.get("eventTypeL2"))
            parts.extend(
                part
                for part in (
                    f"Content: {content}" if content else "",
                    f"Context: {context}" if context else "",
                    f"Tags: {tags}" if projection == "bm25" and tags else "",
                    (
                        f"Event type L1: {event_type_l1}"
                        if projection == "bm25" and event_type_l1
                        else ""
                    ),
                    (
                        f"Event type L2: {event_type_l2}"
                        if projection == "bm25" and event_type_l2
                        else ""
                    ),
                )
                if part
            )
        elif memory_type == "knowledge":
            name = normalize_text(record.get("name"))
            context = normalize_text(record.get("context"))
            parts.extend(
                part
                for part in (
                    f"Name: {name}" if projection == "bm25" and name else "",
                    f"Content: {content}" if content else "",
                    f"Context: {context}" if context else "",
                    f"Tags: {tags}" if projection == "bm25" and tags else "",
                )
                if part
            )
        else:
            # Preserve the original Core projection in both routes.
            if not content and not tags:
                continue
            parts.append(content)
            if tags:
                parts.append(f"Tags: {tags}")

        if len(parts) > 1:
            lines.append(" | ".join(parts))


def mid_bm25_retrieval_text(mid: dict, child_longs: list[dict]) -> str:
    """Lexical Mid document used by BM25 in hybrid seed recall."""
    lines = [
        f"Summary: {normalize_text(mid.get('summary'))}",
        f"Topic: {normalize_text(mid.get('topic_subject'))}",
    ]
    tags = normalize_text(mid.get("tags"))
    if tags:
        lines.append(f"Tags: {tags}")
    _append_typed_child_lines(lines, child_longs, projection="bm25")
    return "\n".join(lines)


def mid_embedding_retrieval_text(mid: dict, child_longs: list[dict]) -> str:
    """Dense Mid document encoded into the enriched embedding sidecar."""
    # Preserve the historical field order so records whose dense projection did not
    # otherwise change (for example Mid + Core only) can reuse their cached vector.
    lines = [
        f"Topic: {normalize_text(mid.get('topic_subject'))}",
        f"Summary: {normalize_text(mid.get('summary'))}",
    ]
    _append_typed_child_lines(lines, child_longs, projection="embedding")
    return "\n".join(lines)


def mid_rerank_retrieval_text(mid: dict, child_longs: list[dict]) -> str:
    """Keep the pre-existing enriched projection used by the seed reranker."""
    lines = [
        f"Topic: {normalize_text(mid.get('topic_subject'))}",
        f"Summary: {normalize_text(mid.get('summary'))}",
    ]
    _append_typed_child_lines(lines, child_longs, projection="rerank")
    return "\n".join(lines)


def enriched_mid_retrieval_text(mid: dict, child_longs: list[dict]) -> str:
    """Backward-compatible name for the text represented by the sidecar vector."""
    return mid_embedding_retrieval_text(mid, child_longs)


def relation_evidence_bundle(mid: dict, child_longs: list[dict]) -> str:
    """Fact-rich but structured input for long-grounded mid relation inference."""
    lines = [
        f"mid_id: {normalize_text(mid.get('id'))}",
        f"session_id: {normalize_text(mid.get('session_id'))}",
        f"session_date: {normalize_text(mid.get('session_date'))}",
        f"topic: {normalize_text(mid.get('topic_subject'))}",
        f"summary: {normalize_text(mid.get('summary'))}",
        "child_long_memories:",
    ]
    if not child_longs:
        lines.append("  (none)")
        return "\n".join(lines)
    for record in child_longs:
        memory_type = normalize_text(record.get("type")) or "unknown"
        fields = [
            f"  - long_id: {normalize_text(record.get('id'))}",
            f"    type: {memory_type}",
            f"    content: {normalize_text(record.get('content'))}",
        ]
        tags = normalize_text(record.get("tags"))
        if tags:
            fields.append(f"    tags: {tags}")
        if memory_type == "episodic":
            for key, label in (
                ("context", "context"),
                ("eventTime", "event_time"),
                ("mentionTime", "mention_time"),
            ):
                value = normalize_text(record.get(key))
                if value:
                    fields.append(f"    {label}: {value}")
        elif memory_type == "knowledge":
            context = normalize_text(record.get("context"))
            if context:
                fields.append(f"    context: {context}")
        lines.extend(fields)
    return "\n".join(lines)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "TYPED_LONG_TABLES",
    "enriched_mid_retrieval_text",
    "group_typed_longs_by_mid",
    "mid_bm25_retrieval_text",
    "mid_embedding_retrieval_text",
    "mid_rerank_retrieval_text",
    "relation_evidence_bundle",
    "text_sha256",
]
