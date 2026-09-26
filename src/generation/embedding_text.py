
from __future__ import annotations


def long_retrieval_content(record: dict) -> str:
    """Build the shared typed-long text for embedding, BM25, rerank, and answering."""
    content = str(record.get("content") or "").strip()
    if not content:
        return ""
    memory_type = str(record.get("type") or "").strip().lower()
    user_id = str(record.get("user_id") or "").strip()
    parts: list[str] = []
    if memory_type == "core":
        entity_name = str(record.get("entityName") or "").strip()
        fact_subject = user_id if entity_name.casefold() == "user" and user_id else entity_name
        parts.append(f"Fact subject: {fact_subject or 'unknown'}")
    parts.extend(
        [
            f"Memory owner: {user_id or 'unknown'}",
            f"Content: {content}",
        ]
    )
    if memory_type == "episodic":
        context = str(record.get("context") or "").strip()
        if context:
            parts.append(f"Context: {context}")
        parts.extend(
            [
                f"Event time: {str(record.get('eventTime') or 'unknown').strip()}",
                f"Mention time: {str(record.get('mentionTime') or 'unknown').strip()}",
            ]
        )
    elif memory_type == "knowledge":
        name = str(record.get("name") or "").strip()
        if name:
            # Keep the compact knowledge-point identifier visible to embedding,
            # lexical recall, reranking, and the final answer model.
            parts.insert(-1, f"Name: {name}")
    return "\n".join(parts)
