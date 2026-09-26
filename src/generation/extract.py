
from __future__ import annotations

import time

from llm import get_response
from llm import prompts

from .util import new_id, now, parse_json

_DEFAULT_CONFIDENCE = 1.0
_STRICT_RESPONSE_ATTEMPTS = 3
_STRICT_RETRY_BACKOFF_SECONDS = 0.2


def _strict_mid_groups(response: str | None, valid_users: set[str]) -> list[dict]:
    if not str(response or "").strip():
        raise RuntimeError("mid extraction model returned no content")
    data = parse_json(response)
    if data is None:
        raise RuntimeError("mid extraction model returned invalid JSON")
    if not isinstance(data, dict):
        raise RuntimeError("mid extraction model returned invalid schema: expected a JSON object")
    if "chat_group" not in data or not isinstance(data["chat_group"], list):
        raise RuntimeError(
            "mid extraction model returned invalid schema: expected a chat_group array"
        )
    groups = data["chat_group"]
    for index, group in enumerate(groups):
        prefix = f"mid extraction model returned invalid schema: chat_group[{index}]"
        if not isinstance(group, dict):
            raise RuntimeError(f"{prefix} must be an object")
        user_id = group.get("user_id")
        if not isinstance(user_id, str) or user_id not in valid_users:
            raise RuntimeError(f"{prefix}.user_id must name a participant")
        if not isinstance(group.get("chat_ids"), list):
            raise RuntimeError(f"{prefix}.chat_ids must be an array")
        for field in ("topic_subject", "summary"):
            value = group.get(field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"{prefix}.{field} must be a non-empty string")
    return groups


def _strict_mid_groups_with_retries(
    prompt: str,
    valid_users: set[str],
) -> list[dict]:
    last_error: RuntimeError | None = None
    for attempt in range(1, _STRICT_RESPONSE_ATTEMPTS + 1):
        try:
            return _strict_mid_groups(get_response(prompt), valid_users)
        except RuntimeError as exc:
            last_error = exc
            if attempt < _STRICT_RESPONSE_ATTEMPTS:
                time.sleep(_STRICT_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def extract_mid_memories(
    *,
    session_history: str,
    conversation_id: str,
    session_id: str,
    session_date: str,
    participants: list[str],
    current_time: str | None = None,
    require_valid_response: bool = True,
    prompt_name: str = "mid_extraction",
) -> list[dict]:
    """Extract Mid records from dialogue formatted by the input adapter.

    ``participants`` restricts valid user IDs. ``current_time`` resolves relative
    dates and defaults to ``session_date``. Malformed responses get up to three
    attempts; response validation cannot be disabled.
    """
    valid_users = set(participants)
    prompt = prompts.render(
        prompt_name,
        CURRENT_TIME=current_time or session_date,
        PARTICIPANTS=", ".join(participants),
        SESSION_HISTORY=session_history,
    )
    if not require_valid_response:
        raise ValueError("Formal Mid extraction requires strict response validation")
    groups = _strict_mid_groups_with_retries(prompt, valid_users)

    memories: list[dict] = []
    for group in groups:
        user_id = group.get("user_id")
        if user_id not in valid_users:
            continue
        memories.append(
            {
                "id": new_id(),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "session_date": session_date,
                "chat_ids": group.get("chat_ids") or [],
                "topic_subject": group.get("topic_subject"),
                "summary": group.get("summary"),
                "tags": group.get("tags") or [],
                **{
                    key: group[key]
                    for key in (
                        "intent_primary",
                        "intent_secondary",
                        "intent_description",
                        "dialogue_phase",
                        "user_attitude",
                    )
                    if key in group
                },
                "confidence": _DEFAULT_CONFIDENCE,
                "created_at": now(),
            }
        )
    if prompt_name == "mid_extraction_beam":
        for record in memories:
            if (
                not isinstance(record.get("dialogue_phase"), str)
                or not record["dialogue_phase"].strip()
            ):
                raise RuntimeError("BEAM Mid extraction requires dialogue_phase")
            attitude = record.get("user_attitude")
            if not isinstance(attitude, dict) or not {"satisfaction_level", "reasoning"} <= set(
                attitude
            ):
                raise RuntimeError("BEAM Mid extraction requires complete user_attitude")
    return memories
