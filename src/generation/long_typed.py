
from __future__ import annotations

from collections.abc import Iterable
import time

from llm import get_response, prompts

from .util import new_id, now, parse_json

# (prompt name, response key, memory type).
_TYPE_PROMPTS = (
    ("memory_extraction_core", "CORE_MEMORY", "core"),
    ("memory_extraction_episodic", "EPISODIC_MEMORY", "episodic"),
    ("memory_extraction_knowledge", "KNOWLEDGE_MEMORY", "knowledge"),
)

# Canonical extraction order.
TYPED_LONG_TYPES = tuple(mtype for _, _, mtype in _TYPE_PROMPTS)
_STRICT_RESPONSE_ATTEMPTS = 3
_STRICT_RETRY_BACKOFF_SECONDS = 0.2


def resolve_typed_long_memory_types(
    memory_types: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Validate a requested type subset and return it in canonical extraction order."""
    if memory_types is None:
        return TYPED_LONG_TYPES
    if isinstance(memory_types, str):
        raise TypeError("memory_types must be an iterable of type names, not a string")
    requested = tuple(str(value).strip().casefold() for value in memory_types)
    if not requested:
        raise ValueError("at least one typed-long memory type must be selected")
    unknown = sorted(set(requested) - set(TYPED_LONG_TYPES))
    if unknown:
        raise ValueError(f"unsupported typed-long memory types: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("typed-long memory types must not contain duplicates")
    requested_set = set(requested)
    return tuple(memory_type for memory_type in TYPED_LONG_TYPES if memory_type in requested_set)


def normalize_tags(value: object, *, limit: int = 5) -> list[str]:
    """Return concise, stable model tags as a de-duplicated string list."""
    if limit < 0:
        raise ValueError("tag limit cannot be negative")
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        tag = " ".join(item.split())
        key = tag.casefold()
        if not tag or key in seen:
            continue
        seen.add(key)
        tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def _prompt_variables(name: str, dialogue: str) -> dict[str, str]:
    """Fill only the placeholders a given prompt declares (see each .txt file)."""
    variables = {"USER_INPUT": dialogue}
    if name == "memory_extraction_core":
        variables["CORE_EXISTING_TOPICS_AND_SUBTOPICS"] = "(none)"
    return variables


def _records(data: object, output_key: str) -> list[dict]:
    if not isinstance(data, dict):
        return []
    items = data.get(output_key)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _strict_type_data(
    response: str | None,
    *,
    output_key: str,
    memory_type: str,
) -> dict:
    if not str(response or "").strip():
        raise RuntimeError(f"{memory_type} typed-long extraction model returned no content")
    data = parse_json(response)
    if data is None:
        raise RuntimeError(f"{memory_type} typed-long extraction model returned invalid JSON")
    if not isinstance(data, dict):
        raise RuntimeError(
            f"{memory_type} typed-long extraction model returned invalid schema: "
            "expected a JSON object"
        )
    if output_key not in data or not isinstance(data[output_key], list):
        raise RuntimeError(
            f"{memory_type} typed-long extraction model returned invalid schema: "
            f"expected a {output_key} array"
        )
    for index, item in enumerate(data[output_key]):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"{memory_type} typed-long extraction model returned invalid schema: "
                f"{output_key}[{index}] must be an object"
            )
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            raise RuntimeError(
                f"{memory_type} typed-long extraction model returned invalid schema: "
                f"{output_key}[{index}].content must be a non-empty string"
            )
    return data


def _strict_type_data_with_retries(
    prompt: str,
    *,
    output_key: str,
    memory_type: str,
) -> dict:
    last_error: RuntimeError | None = None
    for attempt in range(1, _STRICT_RESPONSE_ATTEMPTS + 1):
        try:
            return _strict_type_data(
                get_response(prompt),
                output_key=output_key,
                memory_type=memory_type,
            )
        except RuntimeError as exc:
            last_error = exc
            if attempt < _STRICT_RESPONSE_ATTEMPTS:
                time.sleep(_STRICT_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def _extract_type(
    mid: dict,
    *,
    prompt_name: str,
    output_key: str,
    memory_type: str,
    dialogue: str,
    require_valid_response: bool = True,
) -> list[dict]:
    prompt = prompts.render(
        prompt_name,
        **_prompt_variables(prompt_name, dialogue),
    )
    if not require_valid_response:
        raise ValueError("Formal typed extraction requires strict response validation")
    data = _strict_type_data_with_retries(
        prompt,
        output_key=output_key,
        memory_type=memory_type,
    )
    records: list[dict] = []
    for raw in _records(data, output_key):
        record = dict(raw)  # keep every model field; schemas differ per type
        if memory_type == "core":
            # Core records require tags, including an empty list when none are extracted.
            record["tags"] = normalize_tags(record.get("tags"))
        record["id"] = new_id()
        record["mid_id"] = mid["id"]
        record["user_id"] = mid.get("user_id")
        record["type"] = memory_type
        record["created_at"] = now()
        records.append(record)
    return records


def extract_typed_long_memories(
    mid: dict,
    dialogue: str | None = None,
    *,
    require_valid_response: bool = True,
    memory_types: Iterable[str] | None = None,
) -> dict[str, list[dict]]:
    """Extract typed long memories from a Mid's selected source dialogue.

    Prompts receive raw dialogue only, never the Mid's generated summary, tags, or
    IDs. Each selected type gets a separate request; malformed responses get up to
    three attempts. Validation cannot be disabled.

    Return every type in ``TYPED_LONG_TYPES``, with empty lists for unselected or
    empty types. Preserve model fields and add provenance keys.
    """
    if not isinstance(dialogue, str) or not dialogue.strip():
        raise ValueError("Typed-long extraction requires non-empty raw source dialogue")
    user_input = dialogue
    selected_types = set(resolve_typed_long_memory_types(memory_types))

    result: dict[str, list[dict]] = {memory_type: [] for memory_type in TYPED_LONG_TYPES}
    for name, output_key, mtype in _TYPE_PROMPTS:
        if mtype not in selected_types:
            continue
        result[mtype] = _extract_type(
            mid,
            prompt_name=name,
            output_key=output_key,
            memory_type=mtype,
            dialogue=user_input,
            require_valid_response=require_valid_response,
        )
    return result
