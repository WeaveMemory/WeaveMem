

from __future__ import annotations

import json
import re
from datetime import datetime

import config
from llm import complete_with_usage_details, prompts
from schema.enums import MID_RELATION_TYPES

from .util import new_id, now, parse_json

_SYSTEM = (
    "You are a strict JSON API. Return exactly one JSON object. Do not reveal analysis, "
    "reasoning notes, markdown, or any text outside that object."
)
_MAX_TOKENS = 700
HIGH_CERTAINTY_MIN_CONFIDENCE = 0.95
_REQUIRED_CERTAINTY_CHECKS = (
    "fact_grounded",
    "same_relation_subject",
    "temporal_consistency",
    "direction_correct",
    "no_reasonable_alternative",
)
_EVENT_TIME_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?")


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _string_list(value: object) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [
        _clean_text(item)
        for item in values
        if isinstance(item, (str, int, float)) and _clean_text(item)
    ]


def _normalized_tags(node: dict) -> set[str]:
    return {
        _clean_text(item).casefold()
        for item in _string_list(node.get("tags") or node.get("ttags") or [])
    }


def _named_tag_entities(node: dict) -> set[str]:
    """Use deliberately capitalized tags as conservative named-entity signals."""
    return {
        text.casefold()
        for item in _string_list(node.get("tags") or node.get("ttags") or [])
        if (text := _clean_text(item)) and any(character.isupper() for character in text)
    }


def relation_node_projection(node: dict) -> dict:
    """Return the only endpoint fields visible to the relation judge.
    """
    memory_type = _clean_text(node.get("type")).lower()
    projected: dict = {
        "id": _clean_text(node.get("id")),
        "type": memory_type,
        "content": _clean_text(node.get("content")),
        "tags": _string_list(node.get("tags") or node.get("ttags") or []),
        "session_id": _clean_text(node.get("session_id")),
        "session_date": _clean_text(node.get("session_date")),
    }
    if memory_type == "core":
        projected.update(
            {
                "entityName": _clean_text(node.get("entityName")),
                "topic": _clean_text(node.get("topic")),
                "subTopic": _clean_text(node.get("subTopic")),
            }
        )
    elif memory_type == "episodic":
        projected.update(
            {
                "context": _clean_text(node.get("context")),
                "eventTime": _clean_text(node.get("eventTime")),
                "mentionTime": _clean_text(node.get("mentionTime")),
                "participants": node.get("participants") or [],
            }
        )
    elif memory_type == "knowledge":
        projected.update(
            {
                "name": _clean_text(node.get("name")),
                "context": _clean_text(node.get("context")),
            }
        )
    return projected


def _validated_payload(response: str | None) -> dict:
    data = parse_json(response)
    stripped = str(response or "").strip()
    json_candidate = stripped
    if not isinstance(data, (dict, list)) and stripped.startswith("```"):
        json_candidate = stripped[3:]
        if json_candidate[:4].casefold() == "json":
            json_candidate = json_candidate[4:]
        json_candidate = json_candidate.rstrip()
        if json_candidate.endswith("```"):
            json_candidate = json_candidate[:-3]
        json_candidate = json_candidate.strip()
        data = parse_json(json_candidate)
    # Some endpoints emit the invalid JSON escape \'. Repair only that case.
    if not isinstance(data, (dict, list)) and "\\'" in json_candidate:
        data = parse_json(json_candidate.replace("\\'", "'"))
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        preview = " ".join(str(response or "").split())[:200]
        raise ValueError(f"long-relation pair output must be one JSON object; preview={preview!r}")
    return data


def _event_start(node: dict) -> datetime | None:
    """Return a comparable start time when an endpoint has explicit eventTime."""
    match = _EVENT_TIME_RE.search(_clean_text(node.get("eventTime")))
    if match is None:
        return None
    value = match.group(1)
    if match.group(2):
        value = f"{value}T{match.group(2)}"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _contradiction_confuses_distinct_named_entities(
    source_node: dict,
    target_node: dict,
) -> bool:
    """Reject contradictions that merely compare two named members of one category."""
    source_names = _named_tag_entities(source_node)
    target_names = _named_tag_entities(target_node)
    if not source_names or not target_names or source_names & target_names:
        return False
    source_categories = _normalized_tags(source_node) - source_names
    target_categories = _normalized_tags(target_node) - target_names
    return bool(source_categories & target_categories)


def _has_unverified_project_identity(source_node: dict, target_node: dict) -> bool:
    """Detect generic project-to-project links without a shared specific identifier."""
    source_text = json.dumps(relation_node_projection(source_node), ensure_ascii=False).casefold()
    target_text = json.dumps(relation_node_projection(target_node), ensure_ascii=False).casefold()
    if (
        re.search(r"\bprojects?\b", source_text) is None
        or re.search(r"\bprojects?\b", target_text) is None
    ):
        return False
    generic = {
        "assignment",
        "engineering",
        "engineering project",
        "project",
        "projects",
        "school",
        "task",
        "work",
    }
    generic.update(
        value
        for node in (source_node, target_node)
        for value in (
            _clean_text(node.get("user_id")).casefold(),
            _clean_text(node.get("entityName")).casefold(),
        )
        if value
    )
    shared_specific_tags = (_normalized_tags(source_node) & _normalized_tags(target_node)) - generic
    return not shared_specific_tags


def _passes_certainty_gate(
    data: dict,
    relation_type: str,
    source_node: dict,
    target_node: dict,
    *,
    min_confidence: float,
) -> bool:
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return False
    if confidence < min_confidence:
        return False

    checks = data.get("certainty_checks")
    if not isinstance(checks, dict) or any(
        checks.get(name) is not True for name in _REQUIRED_CERTAINTY_CHECKS
    ):
        return False

    if relation_type in {"causes", "precedes", "enables", "updates"}:
        source_time = _event_start(source_node)
        target_time = _event_start(target_node)
        if source_time is not None and target_time is not None and source_time > target_time:
            return False
    if relation_type == "contradicts" and _contradiction_confuses_distinct_named_entities(
        source_node, target_node
    ):
        return False
    if relation_type in {
        "causes",
        "contradicts",
        "precedes",
        "updates",
    } and _has_unverified_project_identity(source_node, target_node):
        return False
    return True


def judge_projectable_typed_long_pair(
    conversation_id: str,
    node_a: dict,
    node_b: dict,
    *,
    min_confidence: float = HIGH_CERTAINTY_MIN_CONFIDENCE,
) -> tuple[dict | None, dict[str, int]]:
    """Judge one candidate pair and return ``(edge_or_none, token_usage)``."""
    if not HIGH_CERTAINTY_MIN_CONFIDENCE <= min_confidence <= 1.0:
        raise ValueError(
            "long-relation minimum confidence must be between "
            f"{HIGH_CERTAINTY_MIN_CONFIDENCE} and 1"
        )
    node_ids = {_clean_text(node_a.get("id")), _clean_text(node_b.get("id"))}
    if "" in node_ids or len(node_ids) != 2:
        raise ValueError("long-relation pair requires two distinct non-empty node ids")
    for node in (node_a, node_b):
        if node.get("conversation_id") not in (None, "", conversation_id):
            raise ValueError("long-relation pair endpoints must belong to one conversation")
    mid_ids = {_clean_text(node_a.get("mid_id")), _clean_text(node_b.get("mid_id"))}
    if "" in mid_ids or len(mid_ids) != 2:
        raise ValueError("long-relation pair endpoints must belong to different mid blocks")
    session_ids = {
        _clean_text(node_a.get("session_id")),
        _clean_text(node_b.get("session_id")),
    }
    session_id = next(iter(session_ids)) if len(session_ids) == 1 else ""

    prompt = prompts.render(
        "long_relation_projectable_pair",
        CONVERSATION_ID=conversation_id,
        SESSION_ID=session_id,
        USER_ID="",
        USER_IDS=json.dumps(
            list(
                dict.fromkeys(
                    _clean_text(node.get("user_id"))
                    for node in (node_a, node_b)
                    if _clean_text(node.get("user_id"))
                )
            ),
            ensure_ascii=False,
        ),
        NODE_A_OWNER_USER_ID=_clean_text(node_a.get("user_id")),
        NODE_B_OWNER_USER_ID=_clean_text(node_b.get("user_id")),
        MIN_CONFIDENCE=f"{min_confidence:.2f}",
        NODE_A=json.dumps(relation_node_projection(node_a), ensure_ascii=False, sort_keys=True),
        NODE_B=json.dumps(relation_node_projection(node_b), ensure_ascii=False, sort_keys=True),
    )
    response, usage = complete_with_usage_details(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=config.EXTRACTION_MODEL,
        max_tokens=_MAX_TOKENS,
    )
    if response is None:
        raise RuntimeError("long-relation pair model request failed")
    data = _validated_payload(response)
    if data.get("related") is False:
        return None, usage
    if data.get("related") is not True:
        raise ValueError("long-relation pair output requires a boolean related field")

    source_id = _clean_text(data.get("source_id"))
    target_id = _clean_text(data.get("target_id"))
    relation_type = _clean_text(data.get("relation_type"))
    description = _clean_text(data.get("description"))
    source_evidence = _clean_text(data.get("source_evidence"))
    target_evidence = _clean_text(data.get("target_evidence"))
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("long-relation confidence must be numeric") from exc
    if {source_id, target_id} != node_ids:
        raise ValueError("long-relation output must connect exactly the input pair")
    if relation_type not in MID_RELATION_TYPES:
        raise ValueError("long-relation output contains an invalid relation_type")
    if not description or not source_evidence or not target_evidence:
        raise ValueError("long-relation output requires description and endpoint evidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("long-relation confidence must be between 0 and 1")

    node_by_id = {
        _clean_text(node_a["id"]): node_a,
        _clean_text(node_b["id"]): node_b,
    }
    if not _passes_certainty_gate(
        data,
        relation_type,
        node_by_id[source_id],
        node_by_id[target_id],
        min_confidence=min_confidence,
    ):
        return None, usage
    source_mid_id = node_by_id[source_id].get("mid_id")
    target_mid_id = node_by_id[target_id].get("mid_id")
    source_user_id = _clean_text(node_by_id[source_id].get("user_id")) or None
    target_user_id = _clean_text(node_by_id[target_id].get("user_id")) or None
    edge = {
        "id": new_id(),
        "conversation_id": conversation_id,
        "user_id": (source_user_id if source_user_id == target_user_id else None),
        "source_user_id": source_user_id,
        "target_user_id": target_user_id,
        "mid_id": source_mid_id if source_mid_id == target_mid_id else None,
        "source_mid_id": source_mid_id,
        "target_mid_id": target_mid_id,
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": relation_type,
        "description": description,
        "source_evidence": source_evidence,
        "target_evidence": target_evidence,
        "confidence": confidence,
        "created_at": now(),
    }
    return edge, usage
