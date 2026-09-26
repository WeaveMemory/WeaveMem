"""Input adapters: load a conversation corpus into the shape the pipeline expects."""

from .conversations import (
    format_dialogue_for_ids,
    format_session_history,
    iter_sessions,
    load_samples,
    normalize_date_time,
    participants,
)

__all__ = [
    "format_dialogue_for_ids",
    "format_session_history",
    "iter_sessions",
    "load_samples",
    "normalize_date_time",
    "participants",
]
