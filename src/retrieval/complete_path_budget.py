
from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any
import config
import math
from .rerank import rerank_scored

ScoredReranker = Callable[..., list[tuple[dict, float]]]


def _clamp_unit(value: object, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, numeric))


def complete_path_budget_candidates(
    selected_paths: Iterable[dict], seed_items: Iterable[dict]
) -> list[dict]:
    """Combine expanded paths with synthetic zero-hop seed paths."""
    candidates: list[dict] = []
    seen_traversal_paths: set[tuple[str, ...]] = set()
    for index, path in enumerate(selected_paths, start=1):
        node_ids = tuple((str(node_id) for node_id in path.get("node_ids") or []))
        traversal_path = tuple(
            (str(part) for part in path.get("path") or path.get("node_ids") or [])
        )
        if not node_ids or traversal_path in seen_traversal_paths:
            continue
        seen_traversal_paths.add(traversal_path)
        candidate_origin = str(path.get("candidate_origin") or "controller_selected_path")
        candidates.append(
            {
                **path,
                "_budget_candidate_id": f"selected:{index}:{path.get('path_id')}",
                "node_ids": list(node_ids),
                "candidate_origin": candidate_origin,
            }
        )
    for index, seed in enumerate(seed_items, start=1):
        seed_id = str(seed.get("id") or "")
        traversal_path = (seed_id,)
        if not seed_id or traversal_path in seen_traversal_paths:
            continue
        seen_traversal_paths.add(traversal_path)
        candidates.append(
            {
                "_budget_candidate_id": f"seed:{index}:{seed_id}",
                "path_id": f"seed{index}-hop0-budget-candidate",
                "seed_id": seed_id,
                "hop_count": 0,
                "path": [seed_id],
                "node_ids": [seed_id],
                "steps": [],
                "value": seed.get("rerank_score"),
                "reason": "Direct seed evidence retained as a complete zero-hop path.",
                "candidate_origin": "reranked_seed_zero_hop_path",
            }
        )
    return candidates


def score_and_select_complete_paths(
    question: str,
    candidates: Iterable[dict],
    memory_by_id: dict[str, dict],
    *,
    max_nodes: int,
    preferred_relation_types: Iterable[str] = (),
    path_score_weights: tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1),
    rerank_scored_fn: ScoredReranker = rerank_scored,
) -> dict[str, Any]:
    """Score complete paths and greedily retain whole bundles under a unique-node cap."""
    max_nodes = int(max_nodes)
    if max_nodes <= 0:
        raise ValueError("complete-path node budget must be positive")
    if len(path_score_weights) != 4:
        raise ValueError("path_score_weights must contain w_n, w_t, w_d, w_c")
    try:
        w_n, w_t, w_d, w_c = (float(weight) for weight in path_score_weights)
    except (TypeError, ValueError) as exc:
        raise ValueError("path_score_weights must be numeric") from exc
    if any((not math.isfinite(weight) or weight < 0.0 for weight in (w_n, w_t, w_d, w_c))):
        raise ValueError("path_score_weights must be non-negative")
    if abs(w_n + w_t + w_d + w_c - 1.0) > 1e-09:
        raise ValueError("path_score_weights must sum to 1")
    active_weight_total = w_n + w_t + w_d + w_c
    if active_weight_total <= 0.0:
        raise ValueError("enabled path-score components must have positive total weight")
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("node_ids")
        and all((str(node_id) in memory_by_id for node_id in candidate["node_ids"]))
    ]
    if not valid_candidates:
        raise RuntimeError("cannot enforce complete-path node budget without valid path candidates")

    def node_text(candidate: dict) -> str:
        return "\n\n".join(
            (
                str(memory_by_id[str(node_id)].get("content") or "")
                for node_id in candidate["node_ids"]
            )
        )

    node_scores = {
        candidate["_budget_candidate_id"]: score
        for candidate, score in rerank_scored_fn(
            question,
            valid_candidates,
            node_text,
            len(valid_candidates),
            batch_size=int(config.RERANK_BATCH_SIZE),
        )
    }
    edge_candidates = [candidate for candidate in valid_candidates if candidate.get("steps")]

    def edge_text(candidate: dict) -> str:
        return "\n".join(
            (str(step.get("description") or "") for step in candidate.get("steps") or [])
        )

    edge_scores = (
        {
            candidate["_budget_candidate_id"]: score
            for candidate, score in rerank_scored_fn(
                question,
                edge_candidates,
                edge_text,
                len(edge_candidates),
                batch_size=int(config.RERANK_BATCH_SIZE),
            )
        }
        if edge_candidates
        else {}
    )
    preferred_types = {
        str(value).strip().casefold() for value in preferred_relation_types if str(value).strip()
    }
    scored_candidates: list[dict] = []
    for candidate in valid_candidates:
        relation_types = [
            str(step.get("relation_type") or "").strip().casefold()
            for step in candidate.get("steps") or []
            if str(step.get("relation_type") or "").strip()
        ]
        type_fit = (
            sum((relation_type in preferred_types for relation_type in relation_types))
            / len(relation_types)
            if relation_types
            else 0.5
        )
        candidate_id = candidate["_budget_candidate_id"]
        node_score = _clamp_unit(node_scores[candidate_id])
        description_score = _clamp_unit(edge_scores.get(candidate_id), 0.5)
        controller_value = _clamp_unit(candidate.get("value"), 0.5)
        composite = (
            w_n * node_score + w_t * type_fit + w_d * description_score + w_c * controller_value
        )
        scored_candidates.append(
            {
                **candidate,
                "path_score": round(composite, 9),
                "path_query_relevance": round(node_score, 9),
                "question_type_fit": round(type_fit, 9) if type_fit is not None else None,
                "edge_description_support": round(description_score, 9),
                "controller_path_value": round(controller_value, 9)
                if controller_value is not None
                else None,
            }
        )
    scored_candidates.sort(
        key=lambda candidate: (
            -candidate["path_score"],
            len(candidate["node_ids"]),
            candidate["_budget_candidate_id"],
        )
    )
    selected_paths: list[dict] = []
    selected_node_ids: list[str] = []
    selected_node_set: set[str] = set()
    zero_gain_path_count = 0
    hard_budget_rejection_count = 0
    soft_budget_stop_index: int | None = None
    for candidate_index, candidate in enumerate(scored_candidates):
        if len(selected_node_set) >= max_nodes:
            soft_budget_stop_index = candidate_index
            break
        path_node_ids = [str(node_id) for node_id in candidate["node_ids"]]
        new_node_ids = [node_id for node_id in path_node_ids if node_id not in selected_node_set]
        if not new_node_ids:
            zero_gain_path_count += 1
            continue
        if len(selected_node_set | set(path_node_ids)) > max_nodes:
            hard_budget_rejection_count += 1
            continue
        selected_paths.append(candidate)
        for node_id in path_node_ids:
            if node_id not in selected_node_set:
                selected_node_set.add(node_id)
                selected_node_ids.append(node_id)
    if not selected_paths:
        raise RuntimeError("no complete graph path fit within the final memory budget")
    scores_by_node: dict[str, list[float]] = {}
    path_ids_by_node: dict[str, list[str]] = {}
    for path in selected_paths:
        for node_id in path["node_ids"]:
            node_key = str(node_id)
            scores_by_node.setdefault(node_key, []).append(path["path_score"])
            path_ids_by_node.setdefault(node_key, []).append(str(path["path_id"]))
    node_annotations = {
        node_id: {
            "path_budget_score": max(scores_by_node[node_id]),
            "path_budget_path_ids": path_ids_by_node[node_id],
        }
        for node_id in selected_node_ids
    }
    score_formula = {
        "path_query_relevance": w_n,
        "question_type_fit_soft_prior": w_t,
        "edge_description_support": w_d,
        "controller_path_value": w_c,
    }
    stage = {
        "enabled": True,
        "unit": "complete_path_bundle",
        "node_cap_enabled": True,
        "max_unique_nodes": max_nodes,
        "soft_max_unique_nodes": max_nodes,
        "hard_max_unique_nodes": max_nodes,
        "soft_budget_reached": len(selected_node_ids) >= max_nodes,
        "soft_budget_overflow_count": 0,
        "hard_budget_rejection_count": hard_budget_rejection_count,
        "zero_gain_path_count": zero_gain_path_count,
        "candidate_path_count": len(scored_candidates),
        "selected_path_count": len(selected_paths),
        "selected_unique_node_count": len(selected_node_ids),
        "selection": "greedy_by_composite_score_without_partial_paths",
        "score_order": "raw_path_score_descending_no_new_node_divisor",
        "path_score_threshold_enabled": False,
        "soft_stop_rejected_path_count": len(scored_candidates) - soft_budget_stop_index
        if soft_budget_stop_index is not None
        else 0,
        "question_type_fit_enabled": True,
        "question_type_fit_policy": "soft_prior",
        "controller_path_value_enabled": True,
        "edge_score_relation_metadata_enabled": False,
        "score_formula": score_formula,
        "configured_path_score_weights": {"w_n": w_n, "w_t": w_t, "w_d": w_d, "w_c": w_c},
        "selected_paths": selected_paths,
        "rejected_path_count": len(scored_candidates) - len(selected_paths),
    }
    return {
        "selected_paths": selected_paths,
        "selected_node_ids": selected_node_ids,
        "node_annotations": node_annotations,
        "stage": stage,
    }


__all__ = ["complete_path_budget_candidates", "score_and_select_complete_paths"]
