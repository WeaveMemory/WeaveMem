"""Load LoCoMo-style conversation histories for Mid extraction.

Each sample has a ``sample_id`` and a ``conversation`` object containing speaker
names, ``session_N`` turn lists, and optional ``session_N_date_time`` values.
Turns use ``dia_id``, ``speaker``, and ``text``; optional ``query`` and
``blip_caption`` fields describe images.

Sessions are read in numeric order, including sparse indices. ``session_3`` maps
to session ID ``D3`` and turn IDs such as ``D3:1``. Dates are normalized to
``YYYY-MM-DD HH:mm:ss`` when their format is recognized.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Iterator

_SESSION_RE = re.compile(r"^session_(\d+)$")

_DATE_TIME_FMT = "%I:%M %p on %d %B, %Y"


def normalize_date_time(value: str | None) -> str | None:
    """Normalize session dates to ``YYYY-MM-DD HH:mm:ss``; preserve unknown formats."""
    if not value:
        return value
    try:
        return datetime.strptime(value.strip(), _DATE_TIME_FMT).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def load_samples(path: str) -> list[dict]:
    """Load the conversation sample list from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def participants(sample: dict) -> list[str]:
    """Return the two speaker names of a sample."""
    conv = sample["conversation"]
    return [conv[k] for k in ("speaker_a", "speaker_b") if conv.get(k)]


def iter_sessions(sample: dict) -> Iterator[dict]:
    """Yield ``{session_id, session_date, turns}`` records in numeric session order."""
    conv = sample["conversation"]
    numbers = sorted(
        int(m.group(1))
        for k in conv
        if (m := _SESSION_RE.match(k)) and conv.get(k)
    )
    for n in numbers:
        yield {
            "session_id": f"D{n}",
            "session_date": normalize_date_time(conv.get(f"session_{n}_date_time")),
            "turns": conv[f"session_{n}"],
        }


def _image_note(turn: dict) -> str:
    """Keep an image's search subject and generated caption separately labeled.

    The intended subject may differ from the caption's description of the image.
    """
    parts = []
    if turn.get("query"):
        parts.append(f"subject: {turn['query']}")
    if turn.get("blip_caption"):
        parts.append(f"caption: {turn['blip_caption']}")
    return f" [image | {' | '.join(parts)}]" if parts else ""


def _parse_ts(value: str | None) -> datetime | None:
    """Parse a normalized `YYYY-MM-DD HH:mm:ss` timestamp; None if absent/unparseable."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def _format_turn(t: dict, base: datetime | None, index: int) -> str:
    """Render one turn with an optional timestamp and image note.

    ``index`` is its position in the full session, so timestamps stay consistent
    when rendering either the whole session or a selected subset.
    """
    prefix = f"{(base + timedelta(minutes=index)).strftime('%Y-%m-%d %H:%M:%S')} " if base else ""
    return f"{prefix}[{t.get('dia_id')}] {t.get('speaker')}: {t.get('text') or ''}" + _image_note(t)


def format_session_history(turns: list[dict], start_time: str | None = None) -> str:
    """Render turns in input order as ``[dia_id] speaker: text`` with image notes.

    A valid ``start_time`` adds timestamps spaced one minute apart; absent or
    unparseable values leave the lines without timestamps.
    """
    base = _parse_ts(start_time)
    return "\n".join(_format_turn(t, base, i) for i, t in enumerate(turns))


def format_dialogue_for_ids(
    turns: list[dict], chat_ids: list, start_time: str | None = None
) -> str:
    """Render selected source turns for long-term extraction in their original order.

    With a valid ``start_time``, timestamps use each turn's full-session index,
    matching the text used during Mid extraction.
    """
    base = _parse_ts(start_time)
    wanted = set(chat_ids or [])
    return "\n".join(
        _format_turn(t, base, i)
        for i, t in enumerate(turns)
        if t.get("dia_id") in wanted
    )
