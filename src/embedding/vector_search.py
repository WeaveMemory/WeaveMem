
from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rank_by_cosine(
    query: list[float], items: list[tuple[str, list[float]]]
) -> list[tuple[str, float]]:
    scored = [(item_id, cosine(query, vec)) for item_id, vec in items]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
