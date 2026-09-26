

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:

    memories: list[dict]
    trace: dict


__all__ = ["RetrievalResult"]
