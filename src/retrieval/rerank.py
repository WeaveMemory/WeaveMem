

from __future__ import annotations
import logging
import math
import time
from typing import Callable
import config
from llm import providers

logger = logging.getLogger("retrieval.rerank")


def reciprocal_rank_fusion(rankings: list[list[str]]) -> list[str]:
    """Merge several ranked id lists into one via RRF.

    Each ranking contributes 1 / (RRF_K + rank) to an id's fused score (rank is 0-based).
    Ids absent from a ranking get no contribution from it; an empty ranking adds nothing.
    Returns ids best-first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (config.RRF_K + rank)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


def _result_index(item: object) -> int | None:
    """Read the reranked source index from an SDK result item (dict or object)."""
    if isinstance(item, dict):
        return item.get("index")
    return getattr(item, "index", None)


def _result_score(item: object) -> float | None:
    """Read a relevance score from an SDK result item (dict or object)."""
    if isinstance(item, dict):
        value = item.get("relevance_score", item.get("score"))
    else:
        value = getattr(item, "relevance_score", getattr(item, "score", None))
    try:
        score = float(value) if value is not None else None
        return score if score is not None and math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def _call_rerank(query: str, documents: list[str], api_key: str):
    from dashscope import TextReRank

    return TextReRank.call(
        model=config.RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        api_key=api_key,
    )


def rerank_scored(
    query: str,
    records: list[dict],
    text_of: Callable[[dict], str],
    top_n: int,
    *,
    min_score: float | None = None,
    batch_size: int | None = None,
) -> list[tuple[dict, float]]:
    """Score a complete corpus and return globally ranked ``(record, score)`` pairs.

    This API never falls back to unscored input order. Large corpora
    are scored in batches, then merged by numeric relevance score for the same query.
    """
    if not records or top_n <= 0:
        return []
    endpoint = providers.provider_for(config.RERANK_MODEL).endpoint
    if not endpoint.api_key:
        raise RuntimeError("rerank API key is required for scored reranking")
    size = int(batch_size or len(records))
    if size <= 0:
        raise ValueError("rerank batch_size must be positive")
    scored: list[tuple[int, dict, float]] = []
    attempts = max(1, int(config.RERANK_MAX_RETRIES))
    for start in range(0, len(records), size):
        batch = records[start : start + size]
        documents = [text_of(record) or "" for record in batch]
        response = None
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = _call_rerank(query, documents, endpoint.api_key)
                status = getattr(response, "status_code", None)
                if status != 200:
                    raise RuntimeError(f"rerank status={status}")
                break
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    raise RuntimeError(
                        f"scored rerank batch failed after {attempt} attempt(s)"
                    ) from None
                logger.warning("scored rerank batch failed (attempt %d/%d)", attempt, attempts)
                time.sleep(float(config.RERANK_RETRY_BACKOFF_SECONDS) * attempt)
        if response is None:
            raise RuntimeError("scored rerank returned no response") from last_error
        results = getattr(getattr(response, "output", None), "results", None) or []
        batch_scores: dict[int, float] = {}
        for item in results:
            index = _result_index(item)
            score = _result_score(item)
            if type(index) is not int or score is None or not 0 <= index < len(batch):
                raise RuntimeError("scored rerank response contained an invalid index or score")
            if index in batch_scores:
                raise RuntimeError("scored rerank response contained duplicate document indices")
            batch_scores[index] = score
        if len(batch_scores) != len(batch):
            raise RuntimeError(
                "scored rerank response did not contain a valid score for every document"
            )
        scored.extend(
            ((start + index, batch[index], score) for (index, score) in batch_scores.items())
        )
    scored.sort(key=lambda item: (-item[2], item[0]))
    if min_score is not None:
        scored = [item for item in scored if item[2] >= float(min_score)]
    return [(record, score) for (_, record, score) in scored[:top_n]]
