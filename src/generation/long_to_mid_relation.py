
from __future__ import annotations

from collections import defaultdict

from schema.enums import MID_RELATION_TYPES

from .util import new_id, now


_SYMMETRIC_TYPES = {"contradicts"}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _mid_index(mids: list[dict] | dict[str, dict]) -> dict[str, dict]:
    rows = mids.values() if isinstance(mids, dict) else mids
    indexed = {
        _clean(mid.get("id")): mid
        for mid in rows
        if isinstance(mid, dict) and _clean(mid.get("id"))
    }
    if len(indexed) != len(list(mids.values()) if isinstance(mids, dict) else mids):
        raise ValueError("mid memories must have unique non-empty ids")
    return indexed


def project_long_relations_to_mids(
    long_relations: list[dict],
    mids: list[dict] | dict[str, dict],
) -> list[dict]:
    """Merge long edges by directed ``(source_mid, target_mid, type)``.

    Different relation types between the same mid pair survive. Multiple long edges with
    the same projected key become one traversable mid edge whose description and evidence
    retain every distinct long-memory relation.
    """
    mid_by_id = _mid_index(mids)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, str, str]] = []

    for relation in long_relations:
        if not isinstance(relation, dict):
            continue
        source_mid_id = _clean(relation.get("source_mid_id"))
        target_mid_id = _clean(relation.get("target_mid_id"))
        relation_type = _clean(relation.get("relation_type"))
        description = _clean(relation.get("description"))
        source_long_id = _clean(relation.get("source_id"))
        target_long_id = _clean(relation.get("target_id"))
        if (
            not source_mid_id
            or not target_mid_id
            or source_mid_id == target_mid_id
            or relation_type not in MID_RELATION_TYPES
            or not description
            or not source_long_id
            or not target_long_id
        ):
            continue
        if source_mid_id not in mid_by_id or target_mid_id not in mid_by_id:
            raise ValueError("long relation references an unknown parent mid memory")
        source_mid = mid_by_id[source_mid_id]
        target_mid = mid_by_id[target_mid_id]
        if source_mid.get("conversation_id") != target_mid.get("conversation_id"):
            raise ValueError("a projected mid relation cannot cross conversations")

        swapped = relation_type in _SYMMETRIC_TYPES and source_mid_id > target_mid_id
        if swapped:
            source_mid_id, target_mid_id = target_mid_id, source_mid_id
            source_long_id, target_long_id = target_long_id, source_long_id
            source_evidence = _clean(relation.get("target_evidence"))
            target_evidence = _clean(relation.get("source_evidence"))
        else:
            source_evidence = _clean(relation.get("source_evidence"))
            target_evidence = _clean(relation.get("target_evidence"))
        try:
            confidence = float(relation.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if not 0.0 <= confidence <= 1.0:
            continue

        key = (source_mid_id, target_mid_id, relation_type)
        if key not in grouped:
            order.append(key)
        grouped[key].append(
            {
                "long_relation_id": _clean(relation.get("id")),
                "source_long_id": source_long_id,
                "target_long_id": target_long_id,
                "description": description,
                "source_evidence": source_evidence,
                "target_evidence": target_evidence,
                "confidence": confidence,
            }
        )

    projected: list[dict] = []
    for source_mid_id, target_mid_id, relation_type in order:
        evidence: list[dict] = []
        seen: set[tuple] = set()
        for item in grouped[(source_mid_id, target_mid_id, relation_type)]:
            signature = (
                item["source_long_id"],
                item["target_long_id"],
                item["description"].casefold(),
                item["source_evidence"].casefold(),
                item["target_evidence"].casefold(),
            )
            if signature in seen:
                continue
            seen.add(signature)
            evidence.append(item)
        if not evidence:
            continue
        descriptions = list(dict.fromkeys(item["description"] for item in evidence))
        description = (
            descriptions[0]
            if len(descriptions) == 1
            else " ".join(f"({index}) {text}" for index, text in enumerate(descriptions, start=1))
        )
        source_mid = mid_by_id[source_mid_id]
        target_mid = mid_by_id[target_mid_id]
        source_user_id = _clean(source_mid.get("user_id")) or None
        target_user_id = _clean(target_mid.get("user_id")) or None
        projected.append(
            {
                "id": new_id(),
                "conversation_id": source_mid.get("conversation_id"),
                "user_id": (source_user_id if source_user_id == target_user_id else None),
                "source_user_id": source_user_id,
                "target_user_id": target_user_id,
                "source_id": source_mid_id,
                "target_id": target_mid_id,
                "relation_type": relation_type,
                "description": description,
                "evidence": evidence,
                "evidence_count": len(evidence),
                "source_long_ids": list(dict.fromkeys(item["source_long_id"] for item in evidence)),
                "target_long_ids": list(dict.fromkeys(item["target_long_id"] for item in evidence)),
                "confidence": max(item["confidence"] for item in evidence),
                "projection_method": "accepted_long_relation",
                "created_at": now(),
            }
        )
    return projected


__all__ = ["project_long_relations_to_mids"]
