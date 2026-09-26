

from __future__ import annotations
from collections import Counter
from typing import Callable
import config
import embedding
from .bm25 import BM25, tokenize
from .rerank import reciprocal_rank_fusion


def embedding_hits(
    query_vector: list[float], records: list[dict], top_n: int
) -> list[tuple[dict, float]]:
    """Strictly rank records with stored vectors; never fall back to lexical order."""
    if not records or top_n <= 0:
        return []
    items: list[tuple[str, list[float]]] = []
    by_id: dict[str, dict] = {}
    for record in records:
        record_id = str(record.get("id") or "").strip()
        vector = record.get("embedding")
        if not record_id:
            raise ValueError("embedding-ranked mid-memory requires a non-empty id")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"mid-memory {record_id} has no stored embedding")
        if len(vector) != len(query_vector):
            raise ValueError(f"mid-memory {record_id} embedding dimension does not match the query")
        if record_id in by_id:
            raise ValueError(f"duplicate mid-memory id in embedding corpus: {record_id}")
        by_id[record_id] = record
        items.append((record_id, vector))
    ranked = embedding.rank_by_cosine(query_vector, items)
    return [(by_id[record_id], score) for (record_id, score) in ranked[:top_n]]


def hybrid_seed_hits(
    question: str,
    query_vector: list[float],
    records: list[dict],
    text_of: Callable[[dict], str],
    top_n: int,
) -> tuple[list[tuple[dict, dict]], dict]:
    """Fuse full embedding and positive-score BM25 rankings, then keep RRF Top-N."""
    if not records or top_n <= 0:
        return (
            [],
            {
                "candidate_count": len(records),
                "embedding_ranking_count": 0,
                "bm25_positive_count": 0,
                "top_n": max(0, top_n),
                "selected_count": 0,
                "selected": [],
            },
        )
    dense_hits = embedding_hits(query_vector, records, len(records))
    embedding_ids = [str(record["id"]) for (record, _) in dense_hits]
    embedding_score_by_id = {str(record["id"]): score for (record, score) in dense_hits}
    embedding_rank_by_id = {record_id: rank for (rank, record_id) in enumerate(embedding_ids)}
    bm25 = BM25([tokenize(text_of(record)) for record in records])
    bm25_scores = bm25.scores(tokenize(question))
    bm25_order = sorted(
        (index for (index, score) in enumerate(bm25_scores) if score > 0.0),
        key=lambda index: (-bm25_scores[index], index),
    )
    bm25_ids = [str(records[index]["id"]) for index in bm25_order]
    bm25_score_by_id = {str(records[index]["id"]): bm25_scores[index] for index in bm25_order}
    bm25_rank_by_id = {record_id: rank for (rank, record_id) in enumerate(bm25_ids)}
    rankings = [embedding_ids]
    if bm25_ids:
        rankings.append(bm25_ids)
    fused_ids = reciprocal_rank_fusion(rankings)
    rrf_score_by_id: dict[str, float] = {}
    for ranking in rankings:
        for rank, record_id in enumerate(ranking):
            rrf_score_by_id[record_id] = rrf_score_by_id.get(record_id, 0.0) + 1.0 / (
                int(config.RRF_K) + rank
            )
    by_id = {str(record["id"]): record for record in records}
    selected: list[tuple[dict, dict]] = []
    selected_trace: list[dict] = []
    for record_id in fused_ids[:top_n]:
        metadata = {
            "rrf_score": rrf_score_by_id[record_id],
            "embedding_score": embedding_score_by_id[record_id],
            "embedding_rank": embedding_rank_by_id[record_id] + 1,
            "bm25_score": bm25_score_by_id.get(record_id),
            "bm25_rank": bm25_rank_by_id[record_id] + 1 if record_id in bm25_rank_by_id else None,
        }
        record = by_id[record_id]
        selected.append((record, metadata))
        selected_trace.append(
            {
                "id": record_id,
                "mid_id": record.get("mid_id"),
                "user_id": record.get("user_id"),
                "type": record.get("type"),
                **metadata,
            }
        )
    return (
        selected,
        {
            "candidate_count": len(records),
            "candidate_count_by_type": dict(
                sorted(Counter((str(record.get("type") or "") for record in records)).items())
            ),
            "embedding_ranking_count": len(embedding_ids),
            "bm25_positive_count": len(bm25_ids),
            "bm25_zero_score_policy": "excluded_from_bm25_ranking",
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": int(config.RRF_K),
            "top_n": top_n,
            "selected_count": len(selected),
            "selected": selected_trace,
        },
    )
