
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import config
import jsonio
from generation.long_relation_pair import (
    judge_projectable_typed_long_pair,
    relation_node_projection,
)
from generation.long_to_mid_relation import project_long_relations_to_mids
from retrieval.bm25 import BM25, tokenize


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _normalize_tag(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())


def _node_tags(node: dict) -> set[str]:
    raw = node.get("tags") or node.get("ttags") or []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    return {tag for item in values if (tag := _normalize_tag(item))}


def _lexical_text(node: dict) -> str:
    projection = relation_node_projection(node)
    for metadata_field in ("id", "type", "session_id", "session_date"):
        projection.pop(metadata_field, None)
    return json.dumps(projection, ensure_ascii=False, sort_keys=True)


def _normalized_embedding(node: dict) -> list[float]:
    vector = node.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"typed long memory {node.get('id')} has no embedding")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("typed long embedding must contain finite numbers")
    normalized = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in normalized))
    if norm == 0.0:
        raise ValueError(f"typed long memory {node.get('id')} has a zero embedding")
    return [value / norm for value in normalized]


def build_rrf_candidates(
    conversation_id: str,
    nodes: list[dict],
    *,
    top_k: int = 10,
    lemmatize: bool = False,
) -> list[dict]:
    """Return cross-mid pairs selected by per-node Tags+BM25+Embedding RRF Top-K."""
    if top_k <= 0:
        raise ValueError("RRF candidate top_k must be positive")
    if len(nodes) < 2:
        return []
    ids = [_clean(node.get("id")) for node in nodes]
    mid_ids = [_clean(node.get("mid_id")) for node in nodes]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("typed long nodes must have unique non-empty ids")
    if any(not value for value in mid_ids):
        raise ValueError("typed long nodes must have non-empty mid ids")
    vectors = [_normalized_embedding(node) for node in nodes]
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError("typed long embedding dimensions disagree")
    if any(node.get("conversation_id") != conversation_id for node in nodes):
        raise ValueError("graph candidates cannot cross conversations")
    lexical_texts = [_lexical_text(node) for node in nodes]
    bm25 = BM25([tokenize(text, lemmatize=lemmatize) for text in lexical_texts])
    tag_sets = [_node_tags(node) for node in nodes]
    tag_df = Counter(tag for tags in tag_sets for tag in tags)
    tag_idf = {
        tag: math.log((len(nodes) + 1.0) / (frequency + 1.0)) + 1.0
        for tag, frequency in tag_df.items()
    }
    tag_inverted: dict[str, set[int]] = defaultdict(set)
    for node_index, tags in enumerate(tag_sets):
        for tag in tags:
            tag_inverted[tag].add(node_index)
    index_by_id = {record_id: index for index, record_id in enumerate(ids)}
    pairs: dict[tuple[str, str], dict] = {}

    for index, source_id in enumerate(ids):
        eligible = [
            candidate_index
            for candidate_index in range(len(nodes))
            if candidate_index != index and mid_ids[candidate_index] != mid_ids[index]
        ]
        if not eligible:
            continue
        eligible_set = set(eligible)
        shared_tags_by_index: dict[int, set[str]] = defaultdict(set)
        for tag in tag_sets[index]:
            for candidate_index in tag_inverted[tag] & eligible_set:
                shared_tags_by_index[candidate_index].add(tag)
        tag_scored = sorted(
            (
                sum(tag_idf[tag] for tag in shared_tags),
                ids[candidate_index],
            )
            for candidate_index, shared_tags in shared_tags_by_index.items()
        )
        tag_scored.sort(key=lambda item: (-item[0], item[1]))
        tag_ids = [record_id for _score, record_id in tag_scored]
        tag_score_by_id = dict((record_id, score) for score, record_id in tag_scored)
        tag_rank_by_id = {record_id: rank for rank, record_id in enumerate(tag_ids, start=1)}

        embedding_scored = list(
            (
                sum(left * right for left, right in zip(vectors[index], vectors[j])),
                ids[j],
            )
            for j in eligible
        )
        embedding_scored.sort(key=lambda item: (-item[0], item[1]))
        embedding_ids = [record_id for _score, record_id in embedding_scored]
        embedding_score_by_id = dict((record_id, score) for score, record_id in embedding_scored)
        embedding_rank_by_id = {
            record_id: rank for rank, record_id in enumerate(embedding_ids, start=1)
        }

        bm25_scores = bm25.scores(tokenize(lexical_texts[index], lemmatize=lemmatize))
        bm25_scored = sorted(
            (
                float(bm25_scores[j]),
                ids[j],
            )
            for j in eligible
            if bm25_scores[j] > 0.0
        )
        bm25_scored.sort(key=lambda item: (-item[0], item[1]))
        bm25_ids = [record_id for _score, record_id in bm25_scored]
        bm25_score_by_id = dict((record_id, score) for score, record_id in bm25_scored)
        bm25_rank_by_id = {record_id: rank for rank, record_id in enumerate(bm25_ids, start=1)}

        rankings = []
        if tag_ids:
            rankings.append(tag_ids)
        if bm25_ids:
            rankings.append(bm25_ids)
        rankings.append(embedding_ids)
        rrf_scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, record_id in enumerate(ranking):
                rrf_scores[record_id] = rrf_scores.get(record_id, 0.0) + 1.0 / (60 + rank)
        fused_ids = sorted(rrf_scores, key=lambda value: (-rrf_scores[value], value))

        for rrf_rank, candidate_id in enumerate(fused_ids[:top_k], start=1):
            candidate_index = index_by_id[candidate_id]
            pair_key = tuple(sorted((source_id, candidate_id)))
            record = pairs.setdefault(
                pair_key,
                {
                    "conversation_id": conversation_id,
                    "user_ids": sorted(
                        {
                            _clean(nodes[index_by_id[pair_key[0]]].get("user_id")),
                            _clean(nodes[index_by_id[pair_key[1]]].get("user_id")),
                        }
                        - {""}
                    ),
                    "user_id_a": _clean(nodes[index_by_id[pair_key[0]]].get("user_id")),
                    "user_id_b": _clean(nodes[index_by_id[pair_key[1]]].get("user_id")),
                    "cross_user": (
                        _clean(nodes[index_by_id[pair_key[0]]].get("user_id"))
                        != _clean(nodes[index_by_id[pair_key[1]]].get("user_id"))
                    ),
                    "node_id_a": pair_key[0],
                    "node_id_b": pair_key[1],
                    "mid_id_a": mid_ids[index_by_id[pair_key[0]]],
                    "mid_id_b": mid_ids[index_by_id[pair_key[1]]],
                    "same_session": (
                        nodes[index_by_id[pair_key[0]]].get("session_id")
                        == nodes[index_by_id[pair_key[1]]].get("session_id")
                    ),
                    "shared_tags": sorted(
                        tag_sets[index_by_id[pair_key[0]]] & tag_sets[index_by_id[pair_key[1]]]
                    ),
                    "selected_by": [],
                    "selections": [],
                },
            )
            record["selected_by"] = sorted(set(record["selected_by"]) | {source_id})
            record["selections"].append(
                {
                    "source_id": source_id,
                    "candidate_id": candidate_id,
                    "rrf_rank": rrf_rank,
                    "rrf_score": rrf_scores[candidate_id],
                    "tag_rank": tag_rank_by_id.get(candidate_id),
                    "tag_score": tag_score_by_id.get(candidate_id),
                    "embedding_rank": embedding_rank_by_id[candidate_id],
                    "embedding_score": embedding_score_by_id[candidate_id],
                    "bm25_rank": bm25_rank_by_id.get(candidate_id),
                    "bm25_score": bm25_score_by_id.get(candidate_id),
                }
            )
            # Accessing candidate_index here makes the cross-mid invariant explicit.
            if mid_ids[candidate_index] == mid_ids[index]:
                raise AssertionError("same-mid candidate escaped RRF filtering")
    return sorted(
        pairs.values(),
        key=lambda item: (item["conversation_id"], item["node_id_a"], item["node_id_b"]),
    )


def build_relation_graph(
    mids: list[dict],
    longs: list[dict],
    *,
    output_dir: str | Path,
    candidate_top_n: int = 10,
    min_confidence: float = 0.95,
    bm25_lemmatization: bool = False,
) -> dict:
    """Build and publish a complete graph; interrupted pair work is reusable.

    No graph files or success metrics are published until every candidate has a
    valid first-stage judgment. A valid unrelated response is a completed negative
    decision; transport failures and malformed responses remain unfinished.
    """
    if type(candidate_top_n) is not int or candidate_top_n <= 0:
        raise ValueError("graph candidate_top_n must be a positive integer")
    if isinstance(min_confidence, bool) or not 0.95 <= min_confidence <= 1:
        raise ValueError("graph min_confidence must be in [0.95, 1]")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mid_by_id = {str(mid["id"]): mid for mid in mids}
    if len(mid_by_id) != len(mids):
        raise ValueError("graph Mid IDs must be unique")
    prompt_file = Path(__file__).parent / "llm/prompts/long_relation_projectable_pair.txt"
    settings = {
        "model": config.EXTRACTION_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "candidate_top_n": candidate_top_n,
        "min_confidence": min_confidence,
        "rrf_k": 60,
        "rrf_rank_origin": 0,
        "bm25_lemmatization": bm25_lemmatization,
        "independent_post_verification": False,
        "prompt_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
    }
    input_hash = _digest({"mids": mids, "longs": longs, "settings": settings})
    state_path = output / "relation_graph_manifest.json"
    existing = jsonio.read_json(str(state_path), default={}) or {}
    if existing.get("input_sha256") == input_hash:
        for name, expected in existing["output_sha256"].items():
            path = output / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Completed relation graph changed after publication")
        return jsonio.read_json(str(output / "relation_graph_metrics.json"))

    grouped: dict[str, list[dict]] = {}
    seen_ids = set()
    for row in longs:
        parent = mid_by_id.get(str(row.get("mid_id") or ""))
        if parent is None:
            raise ValueError("Typed Long references an unknown parent Mid")
        node_id = str(row.get("id") or "")
        if not node_id or node_id in seen_ids:
            raise ValueError("Typed Long IDs must be non-empty and unique")
        seen_ids.add(node_id)
        cid = str(parent.get("conversation_id") or "")
        if not cid:
            raise ValueError("Graph Mid must identify its conversation")
        node = {
            **row,
            "conversation_id": cid,
            "user_id": parent.get("user_id") or row.get("user_id"),
            "session_id": parent.get("session_id"),
            "session_date": parent.get("session_date"),
        }
        _normalized_embedding(node)
        grouped.setdefault(cid, []).append(node)

    candidates = []
    nodes_by_id = {}
    for cid, nodes in sorted(grouped.items()):
        nodes.sort(key=lambda node: str(node["id"]))
        nodes_by_id.update({str(node["id"]): node for node in nodes})
        candidates.extend(
            build_rrf_candidates(
                cid,
                nodes,
                top_k=candidate_top_n,
                lemmatize=bm25_lemmatization,
            )
        )

    edges = []
    cache_count = 0
    total_usage: Counter = Counter()
    cache_dir = output / ".relation_pair_cache"
    cache_dir.mkdir(exist_ok=True)
    for candidate in candidates:
        node_a, node_b = (nodes_by_id[candidate[key]] for key in ("node_id_a", "node_id_b"))
        pair_hash = _digest(
            {
                "settings": settings,
                "conversation_id": candidate["conversation_id"],
                "node_a": relation_node_projection(node_a),
                "node_b": relation_node_projection(node_b),
                "parents": [node_a["mid_id"], node_b["mid_id"]],
                "owners": [node_a.get("user_id"), node_b.get("user_id")],
            }
        )
        cache_path = cache_dir / f"{pair_hash}.json"
        cached = jsonio.read_json(str(cache_path), default=None)
        if cached is not None:
            if (
                not isinstance(cached, dict)
                or cached.get("input_sha256") != pair_hash
                or cached.get("result_sha256") != _digest(cached.get("result"))
            ):
                raise ValueError("Relation pair cache is incomplete or changed")
            edge = cached["result"]["edge"]
            usage = cached["result"]["usage"]
            cache_count += 1
        else:
            edge, usage = judge_projectable_typed_long_pair(
                candidate["conversation_id"],
                node_a,
                node_b,
                min_confidence=min_confidence,
            )
            if edge is not None:
                edge = {
                    **edge,
                    "acceptance_policy": "first_stage_high_confidence_no_secondary_review",
                }
            result = {"edge": edge, "usage": usage}
            jsonio.atomic_write_json(
                str(cache_path),
                {
                    "input_sha256": pair_hash,
                    "result": result,
                    "result_sha256": _digest(result),
                },
            )
        if edge is not None:
            edges.append(edge)
        total_usage.update({key: value for key, value in usage.items() if isinstance(value, int)})

    projected = project_long_relations_to_mids(edges, mids)
    metrics = {
        "dry_run": False,
        "mid_count": len(mids),
        "long_count": len(longs),
        "scope_count": len(grouped),
        "candidate_count": len(candidates),
        "judged_candidate_count": len(candidates),
        "failure_count": 0,
        "secondary_review_count": 0,
        "accepted_long_relation_count": len(edges),
        "projected_mid_relation_count": len(projected),
        "cached_pair_count": cache_count,
        "accepted_long_relation_type_counts": dict(
            Counter(edge["relation_type"] for edge in edges)
        ),
        "usage": dict(total_usage),
    }
    for filename, value in (
        ("long_relations.json", edges),
        ("mid_relations.json", projected),
        ("relation_graph_metrics.json", metrics),
    ):
        jsonio.atomic_write_json(str(output / filename), value)
    jsonio.atomic_write_json(
        str(state_path),
        {
            "schema_version": 1,
            "input_sha256": input_hash,
            "settings": settings,
            "output_sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in (
                    "long_relations.json",
                    "mid_relations.json",
                    "relation_graph_metrics.json",
                )
            },
        },
    )
    return metrics
