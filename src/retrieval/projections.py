
from __future__ import annotations


def mid_content(record: dict) -> str:
    """Render only Mid fields that are actually present in the record."""
    field_specs = (
        ("topic_subject", ("topic_subject", "topic")),
        ("summary", ("summary",)),
        ("intent_primary", ("intent_primary",)),
        ("intent_secondary", ("intent_secondary",)),
        ("intent_description", ("intent_description",)),
    )
    lines = []
    for label, aliases in field_specs:
        value = next(
            (
                str(record.get(alias) or "").strip()
                for alias in aliases
                if str(record.get(alias) or "").strip()
            ),
            "",
        )
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def typed_long_content(record: dict) -> str:
    """Build the model-visible text for a typed long-memory record."""
    memory_type = str(record.get("type") or "").strip().lower()
    if memory_type == "episodic":
        return episodic_content(record)
    if memory_type == "knowledge":
        return knowledge_content(record)
    return long_content(record)


def typed_long_answer_content(record: dict) -> str:
    """Build answer-visible typed-long text, including exact source evidence."""
    return answer_content_with_source_evidence(typed_long_content(record), record)


def long_content(record: dict) -> str:
    """Render the complete answer-visible Core-memory field contract.

    ``entityName`` is the subject described by a core memory, while ``user_id`` is the
    participant scope inherited from its owning mid-memory. They deliberately remain
    separate because those values can differ.
    """
    content = str(record.get("content") or "").strip()
    if not content:
        return ""
    entity_name = str(record.get("entity_name") or record.get("entityName") or "").strip()
    user_id = str(record.get("user_id") or "").strip()
    fact_subject = user_id if entity_name.casefold() == "user" and user_id else entity_name
    return "\n".join(
        (
            f"user_id: {user_id or 'unknown'}",
            f"topic: {_text_field(record, 'topic')}",
            f"subtopic: {_text_field(record, 'subtopic', 'subTopic')}",
            f"content: {content}",
            f"entity_name: {fact_subject or 'unknown'}",
        )
    )


def episodic_content(record: dict) -> str:
    """Render the complete answer-visible Episodic-memory field contract."""
    content = str(record.get("content") or "").strip()
    if not content:
        return ""
    return "\n".join(
        (
            f"user_id: {_text_field(record, 'user_id')}",
            f"content: {content}",
            f"context: {_text_field(record, 'context')}",
            f"originalTimeExpression: {_text_field(record, 'originalTimeExpression')}",
            f"timePrecision: {_text_field(record, 'timePrecision')}",
            f"eventTime: {_text_field(record, 'eventTime')}",
            f"mentionTime: {_text_field(record, 'mentionTime')}",
        )
    )


def knowledge_content(record: dict) -> str:
    """Render the complete answer-visible Knowledge-memory field contract."""
    content = str(record.get("content") or "").strip()
    if not content:
        return ""
    return "\n".join((f"name: {_text_field(record, 'name')}", f"content: {content}"))


def answer_content_with_source_evidence(content: str, record: dict) -> str:
    """Append source evidence once to an already-rendered answer projection."""
    base = str(content or "").strip()
    evidence = source_evidence_content(record)
    if not evidence or evidence in base:
        return base
    return "\n".join((part for part in (base, evidence) if part))


def _text_field(record: dict, *names: str) -> str:
    for name in names:
        value = str(record.get(name) or "").strip()
        if value:
            return value
    return "unknown"


def source_evidence_content(record: dict) -> str:
    """Render exact source quotes for answer-time grounding.

    Extraction emits ``sourceEvidence`` as ``[{chatId, quote}, ...]``.  Keep this
    projection separate from the retrieval text so exposing provenance to the answer
    model does not change embedding, BM25, or rerank scores.
    """
    raw_evidence = record.get("sourceEvidence") or record.get("source_evidence") or []
    evidence_items = (
        list(raw_evidence) if isinstance(raw_evidence, (list, tuple)) else [raw_evidence]
    )
    lines: list[str] = []
    for item in evidence_items:
        if isinstance(item, dict):
            chat_id = " ".join(str(item.get("chatId") or item.get("chat_id") or "").split())
            quote = " ".join(
                str(item.get("quote") or item.get("verbatimSpan") or item.get("text") or "").split()
            )
        else:
            chat_id = ""
            quote = " ".join(str(item or "").split())
        parts = []
        if chat_id:
            parts.append(f"chatId: {chat_id}")
        if quote:
            parts.append(f"quote: {quote}")
        if parts:
            lines.append("- " + " | ".join(parts))
    return "\n".join(["sourceEvidence:", *lines]) if lines else ""
