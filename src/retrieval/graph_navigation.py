

from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import threading
import config
import embedding
import jsonio
from generation.mid_long_evidence import (
    TYPED_LONG_TABLES,
    group_typed_longs_by_mid,
    mid_bm25_retrieval_text,
    mid_embedding_retrieval_text,
    mid_rerank_retrieval_text,
    text_sha256,
)
from generation.util import parse_json
from llm import complete_with_usage, prompts
from schema.enums import MID_RELATION_TYPES
from .branch_state import apply_branch_decision, branch_records, edge_cards, new_branch
from .complete_path_budget import complete_path_budget_candidates, score_and_select_complete_paths
from .result import RetrievalResult
from .projections import mid_content, typed_long_answer_content, typed_long_content
from .rerank import rerank_scored
from .seed_hits import hybrid_seed_hits


def _selected_graph_relations(
    selected_paths: list[dict], selected_node_ids: list[str]
) -> list[dict]:
    """Expose original directed edges only when both final Mid endpoints survive."""
    allowed = set(selected_node_ids)
    seen = set()
    relations = []
    for path in selected_paths:
        for step in path.get("steps") or []:
            source = str(
                step.get("relation_source_id") or step.get("source_id") or step.get("from_id") or ""
            )
            target = str(
                step.get("relation_target_id") or step.get("target_id") or step.get("to_id") or ""
            )
            relation_type = str(step.get("relation_type") or "").strip().lower()
            description = str(step.get("description") or "").strip()
            if (
                source not in allowed
                or target not in allowed
                or (not relation_type)
                or (not description)
            ):
                continue
            key = str(step.get("edge_id") or "") or (source, target, relation_type, description)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "relation_type": relation_type,
                    "description": description,
                }
            )
    return relations


_QUESTION_TYPE_PRIORS = {
    "when": ["precedes", "updates", "causes"],
    "how": ["causes", "enables", "explains", "precedes"],
    "why": ["explains", "causes", "enables"],
    "what": ["updates", "supports", "contradicts", "precedes"],
    "who": ["supports", "updates", "explains"],
    "where": ["supports", "updates", "precedes"],
    "which": ["supports", "contradicts", "updates"],
    "yes_no": ["supports", "contradicts", "updates"],
    "other": sorted(MID_RELATION_TYPES),
}
_DIRECTION_GUIDANCE = {
    "when": "Use precedes/updates to reconstruct time order. Traverse reverse to find what came before a known event and forward to find what followed; descriptions may override the type prior when they contain the missing date or temporal clue.",
    "how": "Seek mechanism or process: reverse causes/explains/enables from an outcome to its reason or prerequisite, and follow precedes forward through ordered steps.",
    "why": "Seek reasons: usually traverse causes/explains/enables from result to source, but follow the description whenever it explicitly points to the missing reason.",
    "what": "Seek the missing fact or changed state. Prefer updates/supports/contradicts, while allowing any edge whose description concretely predicts the answer.",
    "who": "Use endpoint identity clues in descriptions; relation type is only a soft prior.",
    "where": "Use place clues in descriptions; relation type is only a soft prior.",
    "which": "Use descriptions to distinguish alternatives and verify the requested fact.",
    "yes_no": "Seek confirming, conflicting, or updated evidence before deciding.",
    "other": "Use relation descriptions and direction to follow only answer-bearing paths.",
}
_ENRICHED_EMBEDDING_LOCK = threading.Lock()
_ENRICHED_EMBEDDING_CACHE: tuple[tuple, dict[str, dict]] | None = None


def clear_enriched_embedding_cache() -> None:
    """Drop the cached enriched-embedding sidecar after its file changes."""
    global _ENRICHED_EMBEDDING_CACHE
    with _ENRICHED_EMBEDDING_LOCK:
        _ENRICHED_EMBEDDING_CACHE = None


def _load_enriched_embedding_by_id() -> dict[str, dict]:
    global _ENRICHED_EMBEDDING_CACHE
    path = config.MID_ENRICHED_EMBEDDINGS_PATH
    try:
        stat = os.stat(path)
        signature = (path, stat.st_size, stat.st_mtime_ns)
    except OSError as exc:
        raise ValueError(
            f"enriched mid embeddings are missing at {path}; run the benchmark embed stage"
        ) from exc
    with _ENRICHED_EMBEDDING_LOCK:
        if _ENRICHED_EMBEDDING_CACHE is None or _ENRICHED_EMBEDDING_CACHE[0] != signature:
            rows = jsonio.read_json(path, default=[])
            if not isinstance(rows, list):
                raise ValueError(f"expected a JSON array at {path}")
            _ENRICHED_EMBEDDING_CACHE = (
                signature,
                {
                    str(row.get("id")): row
                    for row in rows
                    if isinstance(row, dict) and str(row.get("id") or "")
                },
            )
        return _ENRICHED_EMBEDDING_CACHE[1]


def _enriched_retrieval_mids(mids: list[dict], scope: dict) -> list[dict]:
    longs_by_mid = group_typed_longs_by_mid(
        {table: list(scope.get(table) or []) for table in TYPED_LONG_TABLES}
    )
    embedding_by_id = _load_enriched_embedding_by_id()
    records: list[dict] = []
    for mid in mids:
        mid_id = str(mid["id"])
        children = longs_by_mid.get(mid_id, [])
        embedding_text = mid_embedding_retrieval_text(mid, children)
        bm25_text = mid_bm25_retrieval_text(mid, children)
        rerank_text = mid_rerank_retrieval_text(mid, longs_by_mid.get(mid_id, []))
        source_mid_id = str(mid.get("source_memory_id") or "").strip()
        sidecar = (
            embedding_by_id.get(mid_id)
            or (embedding_by_id.get(source_mid_id) if source_mid_id else None)
            or {}
        )
        if sidecar.get("embedding_model") != config.EMBEDDING_MODEL:
            raise ValueError(f"mid {mid_id} has no {config.EMBEDDING_MODEL} enriched embedding")
        if sidecar.get("text_sha256") != text_sha256(embedding_text):
            raise ValueError(
                f"mid {mid_id} enriched embedding text is stale; rerun the benchmark embed stage"
            )
        vector = sidecar.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"mid {mid_id} has an invalid enriched embedding")
        records.append(
            {
                **mid,
                "embedding": vector,
                "_bm25_retrieval_text": bm25_text,
                "_embedding_retrieval_text": embedding_text,
                "_enriched_retrieval_text": rerank_text,
            }
        )
    return records


def _mid_bm25_text(record: dict) -> str:
    """Return the enriched lexical projection."""
    return str(record.get("_bm25_retrieval_text") or record.get("_enriched_retrieval_text") or "")


def _query_relevant_child_facts(
    question: str, mid_ids: list[str], scope: dict, *, min_score: float | None, global_limit: int
) -> tuple[dict[str, list[dict]], dict]:
    """Child-fact selection: score the final Mid pool once, threshold, then global Top-K.

    No per-Mid quota is applied. A None threshold preserves all returned scores for
    benchmarks whose recorded protocol disables filtering. Raw non-vector fields
    remain available for benchmark-specific rendering; content uses the frozen view.
    """
    ordered_mid_ids = list(dict.fromkeys((str(mid_id) for mid_id in mid_ids if mid_id)))
    selected_mid_ids = set(ordered_mid_ids)
    threshold = None if min_score is None else float(min_score)
    global_top_n = max(0, int(global_limit))
    longs_by_mid = group_typed_longs_by_mid(
        {table: list(scope.get(table) or []) for table in TYPED_LONG_TABLES}
    )
    candidates = [
        record
        for mid_id in ordered_mid_ids
        for record in longs_by_mid.get(mid_id, [])
        if str(record.get("mid_id") or "") in selected_mid_ids and typed_long_content(record)
    ]
    scored = (
        rerank_scored(
            question,
            candidates,
            typed_long_content,
            len(candidates),
            min_score=None,
            batch_size=int(config.RERANK_BATCH_SIZE),
        )
        if candidates and global_top_n
        else []
    )
    eligible = [
        (record, score)
        for record, score in scored
        if threshold is None or float(score) >= threshold
    ]
    selected_by_mid: dict[str, list[dict]] = defaultdict(list)
    for record, score in eligible[:global_top_n]:
        mid_id = str(record.get("mid_id") or "")
        selected_by_mid[mid_id].append(
            {
                **{key: value for key, value in record.items() if "embedding" not in key},
                "id": str(record.get("id") or ""),
                "mid_id": mid_id,
                "type": str(record.get("type") or ""),
                "score": float(score),
                "mmr_score": None,
                "max_similarity": None,
                "content": typed_long_answer_content(record),
            }
        )
    selected = {
        mid_id: selected_by_mid[mid_id] for mid_id in ordered_mid_ids if selected_by_mid.get(mid_id)
    }
    return (
        selected,
        {
            "enabled": True,
            "candidate_count": len(candidates),
            "candidate_mid_count": sum(
                (bool(longs_by_mid.get(mid_id)) for mid_id in ordered_mid_ids)
            ),
            "rerank_model": config.RERANK_MODEL,
            "min_score": threshold,
            "selection_policy": "global_rerank",
            "global_limit": global_top_n,
            "threshold_eligible_count": len(eligible),
            "selected_count": sum((len(facts) for facts in selected.values())),
            "selected_mid_count": len(selected),
            "limit_saturated": sum((len(facts) for facts in selected.values())) == global_top_n,
            "selected_by_mid": [
                {
                    "mid_id": mid_id,
                    "facts": [
                        {"id": fact["id"], "type": fact["type"], "score": fact["score"]}
                        for fact in facts
                    ],
                }
                for mid_id, facts in selected.items()
            ],
        },
    )


def _mid_content_with_child_facts(record: dict, facts: list[dict]) -> str:
    rendered = [mid_content(record)]
    if not facts:
        return "\n".join(rendered)
    rendered.append("Relevant child facts:")
    for index, fact in enumerate(facts, start=1):
        rendered.append(f"{index}. {fact['content']}")
    return "\n".join(rendered)


def question_strategy(question: str) -> dict:
    """Classify the question and return its relation traversal policy."""
    normalized = " ".join(str(question or "").casefold().split())
    if re.search(
        "\\bwhen\\b|\\bwhat\\s+(?:date|time|day|month|year|age)\\b|\\bhow\\s+(?:long|old|soon|recently)\\b|\\b(before|after|earlier|later)\\b",
        normalized,
    ):
        question_type = "when"
    elif re.search("\\bhow\\b", normalized):
        question_type = "how"
    elif re.search("\\bwhy\\b", normalized):
        question_type = "why"
    elif re.search("\\bwhere\\b", normalized):
        question_type = "where"
    elif re.search("\\bwho(?:se|m)?\\b", normalized):
        question_type = "who"
    elif re.search("\\bwhich\\b", normalized):
        question_type = "which"
    elif re.search("\\bwhat\\b", normalized):
        question_type = "what"
    elif re.match(
        "^(?:did|does|do|is|are|was|were|has|have|had|can|could|will|would)\\b", normalized
    ):
        question_type = "yes_no"
    else:
        question_type = "other"
    return {
        "question_type": question_type,
        "preferred_relation_types": list(_QUESTION_TYPE_PRIORS[question_type]),
        "direction_guidance": _DIRECTION_GUIDANCE[question_type],
        "policy": "soft_type_prior_with_description_override",
    }


def _directional_edge_cards(
    frontier_ids: set[str],
    visited_ids: set[str],
    evaluated_edge_ids: set[str],
    relations: list[dict],
    mid_by_id: dict[str, dict],
    *,
    global_seed_ids: frozenset[str],
) -> list[dict]:
    if global_seed_ids is None:
        raise ValueError("global seed awareness requires an attempt-local seed snapshot")
    cards = edge_cards(frontier_ids, visited_ids, evaluated_edge_ids, relations, mid_by_id)
    enriched: list[dict] = []
    for card in cards:
        traversal_direction = (
            "forward" if card["current_node_id"] == card["source_id"] else "reverse"
        )
        neighbour = mid_by_id[card["neighbour_id"]]
        enriched_card = {
            **card,
            "traversal_direction": traversal_direction,
            "neighbour_summary": neighbour.get("summary") or "",
            "neighbour_already_seed": str(card["neighbour_id"]) in global_seed_ids,
        }
        enriched.append(enriched_card)
    return enriched


def _format_edge_cards(cards: list[dict]) -> str:
    if not cards:
        return "(no unvisited edges are available from this agent's frontier)"
    blocks: list[str] = []
    for card in cards:
        lines = [
            f"edge_id: {card['edge_id']}",
            f"relation_type: {card.get('relation_type') or ''}",
            f"traversal_direction: {card['traversal_direction']}",
            f"description: {card['description']}",
        ]
        if "neighbour_already_seed" in card:
            lines.append(
                f"next_node_already_seed: {str(bool(card['neighbour_already_seed'])).lower()}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_records(records: list[dict]) -> str:
    return "\n\n".join(
        (
            "\n".join(
                [
                    f"memory_id: {record.get('id')}",
                    f"topic: {record.get('topic_subject') or ''}",
                    f"summary: {record.get('summary') or ''}",
                ]
            )
            for record in records
        )
    )


def _parse_navigation_decision(raw: str | None, candidate_edge_ids: set[str]) -> dict | None:
    data = parse_json(raw)
    if not isinstance(data, dict) or set(data) != {"sufficient", "missing", "selected_edges"}:
        return None
    if type(data.get("sufficient")) is not bool:
        return None
    missing = data.get("missing")
    selected = data.get("selected_edges")
    if not isinstance(missing, str) or not isinstance(selected, list):
        return None
    normalized: list[dict] = []
    edge_id_corrections: list[dict] = []
    seen: set[str] = set()
    for item in selected:
        if not isinstance(item, dict) or set(item) != {"edge_id", "reason"}:
            return None
        raw_edge_id = item.get("edge_id")
        reason = item.get("reason")
        if not isinstance(raw_edge_id, str):
            return None
        edge_id = raw_edge_id
        if edge_id not in candidate_edge_ids:
            one_edit_matches = [
                candidate_edge_id
                for candidate_edge_id in candidate_edge_ids
                if raw_edge_id.startswith("midrel-")
                and candidate_edge_id.startswith("midrel-")
                and (len(candidate_edge_id) == len(raw_edge_id))
                and (
                    sum((left != right for left, right in zip(raw_edge_id, candidate_edge_id))) == 1
                )
            ]
            if len(one_edit_matches) != 1:
                return None
            edge_id = one_edit_matches[0]
            edge_id_corrections.append(
                {
                    "raw_edge_id": raw_edge_id,
                    "resolved_edge_id": edge_id,
                    "method": "unique_hamming_distance_1",
                }
            )
        if edge_id in seen or not isinstance(reason, str) or (not reason.strip()):
            return None
        seen.add(edge_id)
        normalized.append({"edge_id": edge_id, "reason": reason.strip()})
    if data["sufficient"] and (missing.strip() or normalized):
        return None
    if not data["sufficient"] and (not missing.strip()):
        return None
    return {
        "sufficient": data["sufficient"],
        "missing": missing.strip(),
        "selected_edge_ids": [item["edge_id"] for item in normalized],
        "selected_edge_reasons": normalized,
        "edge_id_corrections": edge_id_corrections,
    }


def _parse_seed_set_sufficiency(raw: str | None) -> dict | None:
    data = parse_json(raw)
    if not isinstance(data, dict) or set(data) != {"sufficient", "missing"}:
        return None
    if type(data.get("sufficient")) is not bool:
        return None
    missing = data.get("missing")
    if not isinstance(missing, str):
        return None
    missing = missing.strip()
    if data["sufficient"] and missing:
        return None
    if not data["sufficient"] and (not missing):
        return None
    return {"sufficient": data["sufficient"], "missing": missing}


def judge_seed_set_sufficiency(question: str, records: list[dict]) -> dict:
    """Judge the answer-visible seed set once before optional Top-5→Top-10 fallback."""
    if not records:
        raise ValueError("seed-set sufficiency requires at least one memory")
    prompt = prompts.render(
        "mid_graph_seed_set_sufficiency", QUESTION=question, CONTEXT=_format_records(records)
    )
    model = config.GRAPH_NAVIGATION_MODEL
    max_attempts = max(1, int(config.GRAPH_NAVIGATION_MAX_ATTEMPTS))
    total_prompt_tokens = 0
    for attempt in range(1, max_attempts + 1):
        raw, prompt_tokens = complete_with_usage([{"role": "user", "content": prompt}], model=model)
        total_prompt_tokens += int(prompt_tokens or 0)
        decision = _parse_seed_set_sufficiency(raw)
        if decision is not None:
            return {
                **decision,
                "model": model,
                "attempts": attempt,
                "prompt_tokens": total_prompt_tokens,
            }
    raise RuntimeError(
        f"seed-set sufficiency model returned no valid decision after {max_attempts} attempt(s)"
    )


def judge_query_type_navigation(
    question: str,
    strategy: dict,
    records: list[dict],
    edge_cards: list[dict],
    *,
    expansion_hop: int,
    max_hops: int | None = None,
) -> dict:
    if not records:
        raise ValueError("query-type navigation requires at least one memory")
    prompt_name = "mid_graph_query_type_navigation"
    prompt_variables = {
        "QUESTION": question,
        "HOP": expansion_hop,
        "MAX_HOPS": int(config.GRAPH_MAX_HOPS if max_hops is None else max_hops),
        "CONTEXT": _format_records(records),
        "EDGES": _format_edge_cards(edge_cards),
        "NEXT_HOP_INSTRUCTION": "Select using only the relation type, traversal direction, and relation description."
        + (
            " Each edge also marks `next_node_already_seed`. A true value means the next node is already present in this attempt's global seed set, even if this branch has not visited it. Prefer a relevant false edge when it can add missing evidence, but keep a true edge when it supplies ordering, verification, or a necessary bridge to later evidence."
            if any(("neighbour_already_seed" in card for card in edge_cards))
            else ""
        )
        + "",
    }
    prompt_variables["QUESTION_STRATEGY"] = json.dumps(strategy, ensure_ascii=False, sort_keys=True)
    prompt = prompts.render(prompt_name, **prompt_variables)
    candidate_edge_ids = {card["edge_id"] for card in edge_cards}
    model = config.GRAPH_NAVIGATION_MODEL
    max_attempts = max(1, int(config.GRAPH_NAVIGATION_MAX_ATTEMPTS))
    total_prompt_tokens = 0
    for attempt in range(1, max_attempts + 1):
        raw, prompt_tokens = complete_with_usage([{"role": "user", "content": prompt}], model=model)
        total_prompt_tokens += int(prompt_tokens or 0)
        decision = _parse_navigation_decision(raw, candidate_edge_ids)
        if decision is not None:
            return {
                **decision,
                "model": model,
                "attempts": attempt,
                "prompt_tokens": total_prompt_tokens,
            }
    raise RuntimeError(
        f"query-type navigation model returned no valid decision after {max_attempts} attempt(s)"
    )


def _navigation_call(question, strategy, branch, cards, hop, max_hops=None):
    return judge_query_type_navigation(
        question, strategy, branch_records(branch), cards, expansion_hop=hop, max_hops=max_hops
    )


def _path_candidates(
    branch: dict,
    mid_by_id: dict[str, dict],
    edge_by_id: dict[str, dict],
    *,
    global_seed_ids: frozenset[str],
) -> list[dict]:
    if global_seed_ids is None:
        raise ValueError("global seed awareness requires an attempt-local seed snapshot")
    raw_paths = {tuple(path) for paths in branch["paths_by_id"].values() for path in paths}
    ordered_paths = sorted(raw_paths, key=lambda path: (len(path), path))
    seed_awareness_visible = any(len(path) > 1 for path in ordered_paths)
    candidates: list[dict] = []
    for index, path in enumerate(ordered_paths, start=1):
        node_ids = list(path[0::2])
        edge_ids = list(path[1::2])
        steps: list[dict] = []
        for step_index, edge_id in enumerate(edge_ids):
            source_node_id = node_ids[step_index]
            target_node_id = node_ids[step_index + 1]
            edge = edge_by_id[edge_id]
            step = {
                "edge_id": edge_id,
                "from_id": source_node_id,
                "to_id": target_node_id,
                "relation_source_id": str(edge.get("source_id") or ""),
                "relation_target_id": str(edge.get("target_id") or ""),
                "relation_type": edge.get("relation_type"),
                "traversal_direction": "forward"
                if source_node_id == str(edge.get("source_id") or "")
                else "reverse",
                "description": " ".join(str(edge.get("description") or "").split()),
            }
            if seed_awareness_visible:
                step["to_already_seed"] = target_node_id in global_seed_ids
            steps.append(step)
        nodes: list[dict] = []
        for node_id in node_ids:
            node = {
                "id": node_id,
                "topic": mid_by_id[node_id].get("topic_subject") or "",
                "summary": mid_by_id[node_id].get("summary") or "",
            }
            if seed_awareness_visible:
                node["already_seed"] = node_id in global_seed_ids
            nodes.append(node)
        candidate = {
            "path_id": f"seed{branch['seed_rank']}-path{index}",
            "seed_id": branch["seed_id"],
            "hop_count": len(edge_ids),
            "path": list(path),
            "node_ids": node_ids,
            "nodes": nodes,
            "steps": steps,
        }
        if seed_awareness_visible:
            novel_node_ids = [
                node_id
                for node_id in node_ids
                if node_id not in global_seed_ids
            ]
            candidate.update(
                {
                    "novel_node_ids": novel_node_ids,
                    "novel_node_count": len(novel_node_ids),
                    "all_nodes_already_seed": not novel_node_ids,
                }
            )
        candidates.append(candidate)
    return candidates


def _format_paths(candidates: list[dict]) -> str:
    blocks: list[str] = []
    seed_awareness_enabled = any(("novel_node_count" in candidate for candidate in candidates))
    for candidate in candidates:
        lines = [f"path_id: {candidate['path_id']}", f"hop_count: {candidate['hop_count']}"]
        if "novel_node_count" in candidate:
            lines.extend(
                [
                    f"novel_node_count: {candidate['novel_node_count']}",
                    f"all_nodes_already_seed: {str(bool(candidate['all_nodes_already_seed'])).lower()}",
                ]
            )
        lines.append("nodes:")
        for index, node in enumerate(candidate["nodes"]):
            node_line = f"  {index}. {node['id']} | {node['topic']} | {node['summary']}"
            if "already_seed" in node:
                node_line += f" | already_seed={str(bool(node['already_seed'])).lower()}"
            lines.append(node_line)
            if index < len(candidate["steps"]):
                step = candidate["steps"][index]
                step_line = f"     via {step['edge_id']} | {step.get('relation_type') or ''} | {step['traversal_direction']} | {step['description']}"
                if "to_already_seed" in step:
                    step_line += f" | to_already_seed={str(bool(step['to_already_seed'])).lower()}"
                lines.append(step_line)
        blocks.append("\n".join(lines))
    rendered = "\n\n---\n\n".join(blocks)
    if not seed_awareness_enabled:
        return rendered
    return f"Seed-awareness note: `already_seed=true` means the node was already in this attempt's global seed set, even if this branch has not visited it. Prefer relevant paths that add nodes with `already_seed=false`. A path containing only seed nodes remains useful only for ordering or verification; an already-seed intermediate node may still be a necessary bridge to a later novel node.\n\n{rendered}"


def _parse_path_selections(raw: str | None, path_ids: set[str], max_paths: int) -> dict | None:
    data = parse_json(raw)
    if not isinstance(data, dict) or set(data) != {"selected_paths", "reason"}:
        return None
    raw_selections = data.get("selected_paths")
    reason = data.get("reason")
    if (
        not isinstance(raw_selections, list)
        or len(raw_selections) > max_paths
        or (not isinstance(reason, str))
        or (not reason.strip())
    ):
        return None
    selections: list[dict] = []
    selected_ids: set[str] = set()
    for item in raw_selections:
        if not isinstance(item, dict) or set(item) != {"path_id", "value", "reason"}:
            return None
        path_id = item.get("path_id")
        item_reason = item.get("reason")
        if (
            path_id not in path_ids
            or path_id in selected_ids
            or (not isinstance(item_reason, str))
            or (not item_reason.strip())
        ):
            return None
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= value <= 1.0:
            return None
        selected_ids.add(path_id)
        selections.append({"path_id": path_id, "value": value, "reason": item_reason.strip()})
    return {"selected_paths": selections, "reason": reason.strip()}


def judge_valuable_branch_paths(
    question: str, strategy: dict, branch: dict, candidates: list[dict], max_paths: int
) -> dict:
    if not candidates:
        raise ValueError("multi-path selection requires at least one candidate path")
    if max_paths <= 0:
        raise ValueError("multi-path selection requires a positive path limit")
    prompt_name = "mid_graph_query_type_paths_select"
    prompt_variables = {
        "QUESTION": question,
        "SEED_ID": branch["seed_id"],
        "MAX_PATHS": str(max_paths),
        "PATHS": _format_paths(candidates),
    }
    prompt_variables["QUESTION_STRATEGY"] = json.dumps(strategy, ensure_ascii=False, sort_keys=True)
    prompt = prompts.render(prompt_name, **prompt_variables)
    model = config.GRAPH_PATH_MODEL
    max_attempts = max(1, int(config.GRAPH_PATH_MAX_ATTEMPTS))
    total_prompt_tokens = 0
    path_ids = {candidate["path_id"] for candidate in candidates}
    for attempt in range(1, max_attempts + 1):
        raw, prompt_tokens = complete_with_usage([{"role": "user", "content": prompt}], model=model)
        total_prompt_tokens += int(prompt_tokens or 0)
        selection = _parse_path_selections(raw, path_ids, max_paths)
        if selection is not None:
            return {
                **selection,
                "model": model,
                "attempts": attempt,
                "prompt_tokens": total_prompt_tokens,
            }
    raise RuntimeError(
        f"multi-path model returned no valid selection after {max_attempts} attempt(s)"
    )


def _empty_result(question: str, mode: str = "graph") -> RetrievalResult:
    return RetrievalResult(
        memories=[],
        trace={
            "mode": mode,
            "query": question,
            "question_strategy": question_strategy(question),
            "initial_embedding_bm25_rrf_stage": {
                "candidate_count": 0,
                "selected_count": 0,
                "selected": [],
            },
            "react_path_navigation": {
                "seed_agent_count": 0,
                "parallel": True,
                "max_hops": int(config.GRAPH_MAX_HOPS),
                "stop_reason": "empty_mid_scope",
                "branches": [],
            },
            "final": {
                "mid_count": 0,
                "long_count": 0,
                "count": 0,
                "selected": [],
                "prompt_projection": {
                    "mid": "selected best paths: topic_subject + summary -> content",
                    "relation_type_direction_description": "navigation_and_trace_only",
                    "typed_long": "excluded",
                },
            },
        },
    )


def _resolved_parameter_trace(trace: dict, *, complete_path_budget: int) -> dict:
    """Return the parameter vector a post-path run actually resolved to."""
    post_path = trace.get("post_path_sufficiency_fallback") or {}
    configuration = post_path.get("configuration") or {}
    navigation = trace.get("react_path_navigation") or {}
    best_path_stage = trace.get("best_path_stage") or {}
    return {
        "schema_version": 1,
        "max_graph_hops": int(
            configuration.get("max_hops", navigation.get("max_hops", config.GRAPH_MAX_HOPS))
        ),
        "max_paths_per_seed": int(
            configuration.get(
                "max_paths_per_seed",
                best_path_stage.get("max_paths_per_seed_agent", config.GRAPH_MAX_PATHS_PER_SEED),
            )
        ),
        "complete_path_mid_budget": int(complete_path_budget),
        "initial_seed_width": int(
            configuration.get("initial_seed_top_n", config.GRAPH_INITIAL_SEED_TOP_N)
        ),
        "child_fact_top_k": int(config.GRAPH_CHILD_FACT_GLOBAL_TOP_N),
        "child_fact_threshold": config.GRAPH_CHILD_FACT_MIN_SCORE,
        "initial_rrf_candidate_pool": int(
            configuration.get(
                "initial_rrf_candidate_top_n", config.GRAPH_INITIAL_RRF_CANDIDATE_TOP_N
            )
        ),
        "fallback_seed_width": int(
            configuration.get("fallback_seed_top_n", config.GRAPH_FALLBACK_SEED_TOP_N)
        ),
        "fallback_rrf_candidate_pool": int(
            configuration.get(
                "fallback_rrf_candidate_top_n", config.GRAPH_FALLBACK_RRF_CANDIDATE_TOP_N
            )
        ),
    }


def _search_paths(
    question: str,
    scope: dict,
    *,
    top_k: int | None = None,
    mode: str,
    fixed_seed_top_n: int | None = None,
    max_selected_paths_per_seed: int = 1,
    initial_seed_rerank_rrf_candidate_top_n: int | None = None,
    max_hops_override: int | None = None,
) -> RetrievalResult:
    """Run independent seed agents and merge their selected complete paths."""
    del top_k
    mids = [record for record in scope.get("mids") or [] if mid_content(record)]
    if not mids:
        return _empty_result(question, mode)
    query_vector = embedding.embed_text(question)
    if not query_vector:
        raise RuntimeError("query embedding is required for query-type path traversal")
    strategy = question_strategy(question)
    raw_mid_by_id = scope.get("mid_by_id") or {str(record["id"]): record for record in mids}
    mid_by_id = {str(record_id): record for record_id, record in raw_mid_by_id.items()}
    relations = list(scope.get("mid_rels") or [])
    edge_by_id = {str(edge.get("id")): edge for edge in relations if edge.get("id") is not None}
    retrieval_mids = _enriched_retrieval_mids(mids, scope)
    original_by_id = {str(record["id"]): record for record in mids}
    seed_top_n = max(0, int(fixed_seed_top_n))
    initial_rerank_candidate_top_n = max(
        seed_top_n, int(initial_seed_rerank_rrf_candidate_top_n or seed_top_n)
    )
    initial_rerank_trace = {
        "enabled": True,
        "rrf_candidate_top_n": initial_rerank_candidate_top_n,
        "rrf_candidate_count": 0,
        "rerank_top_n": seed_top_n,
        "rerank_input_projection": "topic_subject + summary + child long content/tags",
        "selected_count": 0,
        "selected": [],
    }
    initial_candidates, initial_candidate_trace = hybrid_seed_hits(
        question, query_vector, retrieval_mids, _mid_bm25_text, initial_rerank_candidate_top_n
    )
    initial_candidate_records = [record for record, _metadata in initial_candidates]
    initial_metadata_by_id = {
        str(record["id"]): metadata for record, metadata in initial_candidates
    }
    initial_reranked = rerank_scored(
        question,
        initial_candidate_records,
        lambda record: str(record.get("_enriched_retrieval_text") or ""),
        seed_top_n,
        batch_size=int(config.RERANK_BATCH_SIZE),
    )
    initial_seed_hits = [
        (
            original_by_id[str(record["id"])],
            {
                **initial_metadata_by_id[str(record["id"])],
                "rerank_score": rerank_score,
                "rerank_rank": rank,
            },
        )
        for rank, (record, rerank_score) in enumerate(initial_reranked, start=1)
    ]
    initial_selected = [
        {
            "id": record.get("id"),
            "user_id": record.get("user_id"),
            **initial_metadata_by_id[str(record["id"])],
            "rerank_score": rerank_score,
            "rerank_rank": rank,
        }
        for rank, (record, rerank_score) in enumerate(initial_reranked, start=1)
    ]
    initial_seed_trace = {
        **initial_candidate_trace,
        "top_n": seed_top_n,
        "selected_count": len(initial_seed_hits),
        "selected": initial_selected,
        "retrieval_projection": "split_bm25_embedding_mid_child_fields_v2",
        "bm25_projection": "mid(summary,topic_subject,tags); episodic(content,context,tags,eventTypeL1,eventTypeL2); knowledge(name,content,context,tags); core unchanged",
        "embedding_projection": "mid(summary,topic_subject); episodic/knowledge(content,context); core unchanged",
        "answer_projection": "topic_subject_plus_summary_only",
        "embedding_sidecar": config.MID_ENRICHED_EMBEDDINGS_PATH,
    }
    initial_rerank_trace = {
        **initial_rerank_trace,
        "rrf_candidate_count": len(initial_candidates),
        "selected_count": len(initial_seed_hits),
        "selected": initial_selected,
    }
    if not initial_seed_hits:
        return _empty_result(question, mode)
    seed_hits, final_seed_trace = (initial_seed_hits, initial_seed_trace)
    seed_records = [record for record, _metadata in seed_hits]
    seed_trace = {
        **final_seed_trace,
        "initial_top_n": seed_top_n,
        "initial_selected_count": len(initial_seed_hits),
        "initial_selected": initial_seed_trace.get("selected", []),
        "initial_rrf_rerank": initial_rerank_trace,
        "seed_selection_policy": "fixed_top_n",
    }
    attempt_global_seed_ids = frozenset((str(record["id"]) for record in seed_records))
    branches = []
    for rank, record in enumerate(seed_records, start=1):
        branch = new_branch(record, rank)
        branch["global_seed_ids"] = attempt_global_seed_ids
        branches.append(branch)
    seed_metadata = {str(record["id"]): metadata for record, metadata in seed_hits}
    max_hops = max(
        0, int(config.GRAPH_MAX_HOPS if max_hops_override is None else max_hops_override)
    )
    worker_limit = max(1, int(config.GRAPH_AGENT_WORKERS))
    path_limit = max(1, int(max_selected_paths_per_seed))
    controller_prompt_tokens = 0
    hops_executed = 0
    navigation_hop_limit = max_hops
    for hop in range(1, navigation_hop_limit + 1):
        jobs: list[tuple[dict, list[dict]]] = []
        for branch in branches:
            if not branch["active"]:
                continue
            cards = _directional_edge_cards(
                set(branch["frontier_ids"]),
                set(branch["visited"]),
                set(branch["evaluated_edge_ids"]),
                relations,
                mid_by_id,
                global_seed_ids=branch["global_seed_ids"],
            )
            if not cards:
                branch["active"] = False
                branch["stop_reason"] = "no_candidate_edges"
                branch["rounds"].append(
                    {
                        "expansion_hop": hop,
                        "frontier_node_ids": sorted(branch["frontier_ids"]),
                        "memory_count_before": len(branch["visited"]),
                        "candidate_edge_count": 0,
                        "candidate_edges": [],
                        "decision": None,
                        "selected_edges": [],
                        "expanded_node_ids": [],
                        "memory_count_after": len(branch["visited"]),
                    }
                )
                continue
            jobs.append((branch, cards))
        if not jobs:
            break
        decisions: dict[str, tuple[list[dict], dict]] = {}
        with ThreadPoolExecutor(max_workers=min(len(jobs), worker_limit)) as executor:
            future_map = {
                executor.submit(
                    _navigation_call, question, strategy, branch, cards, hop, max_hops
                ): (branch, cards)
                for branch, cards in jobs
            }
            for future in as_completed(future_map):
                branch, cards = future_map[future]
                decisions[branch["seed_id"]] = (cards, future.result())
        round_progress = False
        for branch, _cards in jobs:
            cards, decision = decisions[branch["seed_id"]]
            controller_prompt_tokens += int(decision.get("prompt_tokens") or 0)
            new_ids, _round_trace = apply_branch_decision(branch, cards, decision, mid_by_id, hop)
            if new_ids:
                round_progress = True
                hops_executed = max(hops_executed, hop)
        if not round_progress and (not any((branch["active"] for branch in branches))):
            break
    for branch in branches:
        if branch["stop_reason"] is None:
            branch["stop_reason"] = "max_hops_reached"
    path_candidates_by_seed = {
        branch["seed_id"]: _path_candidates(
            branch,
            mid_by_id,
            edge_by_id,
            global_seed_ids=branch["global_seed_ids"],
        )
        for branch in branches
    }
    path_selections: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(branches), worker_limit)) as executor:
        future_map = {
            executor.submit(
                judge_valuable_branch_paths,
                question,
                strategy,
                branch,
                path_candidates_by_seed[branch["seed_id"]],
                path_limit,
            ): branch
            for branch in branches
        }
        for future in as_completed(future_map):
            branch = future_map[future]
            path_selections[branch["seed_id"]] = future.result()
    selected_paths: list[dict] = []
    merged_records: dict[str, dict] = {}
    provenance_by_id: dict[str, list[dict]] = {}
    for branch in branches:
        seed_id = branch["seed_id"]
        merged_records.setdefault(seed_id, mid_by_id[seed_id])
        provenance_by_id.setdefault(seed_id, []).append(
            {
                "seed_id": seed_id,
                "seed_rank": branch["seed_rank"],
                "path_id": None,
                "position": 0,
                "hop": 0,
                "path": [seed_id],
                "value": None,
                "reason": "RRF seed retained independently of graph expansion",
            }
        )
    for branch in branches:
        seed_id = branch["seed_id"]
        selection = path_selections[seed_id]
        controller_prompt_tokens += int(selection.get("prompt_tokens") or 0)
        candidate_by_id = {
            candidate["path_id"]: candidate for candidate in path_candidates_by_seed[seed_id]
        }
        selected_items = selection["selected_paths"]
        for selected_item in selected_items:
            selected = {
                **candidate_by_id[selected_item["path_id"]],
                "value": selected_item["value"],
                "reason": selected_item["reason"],
                "selection_set_reason": selection["reason"],
                "model": selection["model"],
                "attempts": selection["attempts"],
                "selection_prompt_tokens": selection["prompt_tokens"],
            }
            selected_paths.append(selected)
            for position, node_id in enumerate(selected["node_ids"]):
                merged_records.setdefault(node_id, mid_by_id[node_id])
                provenance_by_id.setdefault(node_id, []).append(
                    {
                        "seed_id": seed_id,
                        "seed_rank": branch["seed_rank"],
                        "path_id": selected["path_id"],
                        "position": position,
                        "hop": position,
                        "path": selected["path"],
                        "value": selected["value"],
                        "reason": selected["reason"],
                    }
                )
    child_fact_trace = {
        "enabled": False,
        "candidate_count": 0,
        "selected_count": 0,
        "selected_mid_count": 0,
        "selected_by_mid": [],
    }
    memories: list[dict] = []
    for record_id, record in merged_records.items():
        provenance = provenance_by_id[record_id]
        seed_info = seed_metadata.get(record_id, {})
        memories.append(
            {
                **record,
                "content": mid_content(record),
                "memory_pool": "mid",
                "retrieval_origin": "hybrid_rrf_seed"
                if record_id in seed_metadata
                else "query_type_best_path",
                "graph_hop": min((item["hop"] for item in provenance)),
                "graph_path": provenance[0]["path"],
                "graph_paths": provenance,
                "embedding_score": seed_info.get("embedding_score"),
                "bm25_score": seed_info.get("bm25_score"),
                "rrf_score": seed_info.get("rrf_score"),
                "seed_rerank_score": seed_info.get("rerank_score"),
                "seed_rerank_rank": seed_info.get("rerank_rank"),
                "query_relevant_child_facts": [],
            }
        )
    selected_navigation_edges = [
        edge
        for branch in branches
        for round_trace in branch["rounds"]
        for edge in round_trace.get("selected_edges", [])
    ]
    candidate_navigation_edges = [
        edge
        for branch in branches
        for round_trace in branch["rounds"]
        for edge in round_trace.get("candidate_edges", [])
    ]
    final_steps = [step for path in selected_paths for step in path["steps"]]
    branch_traces = []
    for branch in branches:
        seed_id = branch["seed_id"]
        branch_traces.append(
            {
                "seed_id": seed_id,
                "seed_rank": branch["seed_rank"],
                "visited_node_count": len(branch["visited"]),
                "evaluated_edge_count": len(branch["evaluated_edge_ids"]),
                "stop_reason": branch["stop_reason"],
                "rounds": branch["rounds"],
                "candidate_path_count": len(path_candidates_by_seed[seed_id]),
                "path_selection": path_selections[seed_id],
            }
        )
    hop_counts = Counter((memory["graph_hop"] for memory in memories))
    final_counts = {f"hop{hop}_mid_count": hop_counts[hop] for hop in range(6)}
    return RetrievalResult(
        memories=memories,
        trace={
            "mode": mode,
            "query": question,
            "question_strategy": strategy,
            "question_type_guidance_enabled": True,
            "initial_embedding_bm25_rrf_stage": seed_trace,
            "react_path_navigation": {
                "seed_agent_count": len(branches),
                "parallel": True,
                "agents_independent": True,
                "multi_edge_selection": True,
                "path_generation_policy": "glm_react_navigation",
                "controller_navigation_enabled": True,
                "controller_path_selection_enabled": True,
                "simple_path_node_revisit_allowed": False,
                "relation_direction": "stored_direction_preserved",
                "relation_description_used": True,
                "next_hop_content_visible": False,
                "relation_type_policy": "soft_question_type_prior",
                "max_hops": max_hops,
                "hops_executed": hops_executed,
                "controller_prompt_tokens": controller_prompt_tokens,
                "selected_navigation_edge_count": len(selected_navigation_edges),
                "global_seed_awareness": {
                    "snapshot_scope": "active_retrieval_attempt",
                    "seed_count": len(attempt_global_seed_ids),
                    "seed_ids": sorted(attempt_global_seed_ids),
                    "snapshot_container": "frozenset",
                    "worker_access": "shared_read_only",
                    "dynamic_cross_branch_discovery_shared": False,
                    "same_target_policy": "retain_branch_provenance_then_deduplicate_final_nodes_by_mid_id",
                },
                "candidate_next_node_already_seed_count": sum(
                    (
                        bool(edge.get("neighbour_already_seed"))
                        for edge in candidate_navigation_edges
                    )
                ),
                "candidate_next_node_novel_count": sum(
                    (
                        not bool(edge.get("neighbour_already_seed"))
                        for edge in candidate_navigation_edges
                    )
                ),
                "selected_next_node_already_seed_count": sum(
                    (bool(edge.get("neighbour_already_seed")) for edge in selected_navigation_edges)
                ),
                "selected_next_node_novel_count": sum(
                    (
                        not bool(edge.get("neighbour_already_seed"))
                        for edge in selected_navigation_edges
                    )
                ),
                "selected_navigation_relation_type_counts": dict(
                    Counter(
                        (str(edge.get("relation_type") or "") for edge in selected_navigation_edges)
                    )
                ),
                "selected_navigation_direction_counts": dict(
                    Counter(
                        (
                            str(edge.get("traversal_direction") or "")
                            for edge in selected_navigation_edges
                        )
                    )
                ),
                "branches": branch_traces,
            },
            "best_path_stage": {
                "path_selection_protocol": "multi_path_cap",
                "path_generation_policy": "glm_selected_paths",
                "one_path_per_seed_agent": False,
                "max_paths_per_seed_agent": path_limit,
                "seed_retention_policy": "all_rrf_seeds_retained",
                "candidate_path_count": sum(
                    (len(paths) for paths in path_candidates_by_seed.values())
                ),
                "selected_path_count": len(selected_paths),
                "candidate_path_with_novel_node_count": sum(
                    (
                        int(path.get("novel_node_count") or 0) > 0
                        for paths in path_candidates_by_seed.values()
                        for path in paths
                    )
                ),
                "candidate_seed_only_path_count": sum(
                    (
                        int(path.get("novel_node_count") or 0) == 0
                        for paths in path_candidates_by_seed.values()
                        for path in paths
                    )
                ),
                "selected_path_with_novel_node_count": sum(
                    (int(path.get("novel_node_count") or 0) > 0 for path in selected_paths)
                ),
                "selected_seed_only_path_count": sum(
                    (int(path.get("novel_node_count") or 0) == 0 for path in selected_paths)
                ),
                "selected_unique_novel_node_count": len(
                    {
                        str(node_id)
                        for path in selected_paths
                        for node_id in path.get("novel_node_ids") or []
                    }
                ),
                "selected_seed_agent_count": sum(
                    (
                        bool(
                            [
                                path
                                for path in selected_paths
                                if path["seed_id"] == branch["seed_id"]
                            ]
                        )
                        for branch in branches
                    )
                ),
                "selected_path_count_by_seed": {
                    branch["seed_id"]: sum(
                        (path["seed_id"] == branch["seed_id"] for path in selected_paths)
                    )
                    for branch in branches
                },
                "selected_paths": selected_paths,
                "selected_relation_type_counts": dict(
                    Counter((str(step.get("relation_type") or "") for step in final_steps))
                ),
                "selected_direction_counts": dict(
                    Counter((step["traversal_direction"] for step in final_steps))
                ),
            },
            "child_atomic_evidence_stage": child_fact_trace,
            "final_rerank_stage": {
                "enabled": False,
                "candidate_count": len(memories),
                "selected_count": len(memories),
            },
            "final": {
                **final_counts,
                "mid_count": len(memories),
                "core_count": 0,
                "other_count": 0,
                "long_count": 0,
                "child_long_fact_count": int(child_fact_trace.get("selected_count") or 0),
                "mids_with_child_facts": int(child_fact_trace.get("selected_mid_count") or 0),
                "count": len(memories),
                "selected": [record["id"] for record in memories],
                "prompt_projection": {
                    "mid": "all RRF seeds plus selected valuable path nodes: "
                    + "topic_subject + summary -> content",
                    "relation_type_direction_description": "navigation_and_trace_only",
                    "typed_long": "excluded",
                },
            },
        },
    )


def _run_one_path_attempt(
    question: str,
    scope: dict,
    *,
    top_k: int | None,
    mode: str,
    seed_top_n: int,
    rrf_candidate_top_n: int,
    max_hops_override: int | None = None,
) -> RetrievalResult:
    """Run a complete navigation attempt at one seed and candidate width."""
    return _search_paths(
        question,
        scope,
        top_k=top_k,
        mode=mode,
        fixed_seed_top_n=int(seed_top_n),
        initial_seed_rerank_rrf_candidate_top_n=int(rrf_candidate_top_n),
        max_selected_paths_per_seed=int(config.GRAPH_MAX_PATHS_PER_SEED),
        max_hops_override=int(
            config.GRAPH_MAX_HOPS if max_hops_override is None else max_hops_override
        ),
    )


def _tag_post_path_attempt(result: RetrievalResult, *, attempt_id: str) -> RetrievalResult:
    """Attach an attempt namespace to every path-bearing audit record."""
    trace = result.trace
    seed_stage = trace.get("initial_embedding_bm25_rrf_stage") or {}
    tagged_seed_stage = {
        **seed_stage,
        "attempt_id": attempt_id,
        "selected": [
            {**item, "attempt_id": attempt_id} for item in seed_stage.get("selected") or []
        ],
    }
    navigation = trace.get("react_path_navigation") or {}
    tagged_branches: list[dict] = []
    for branch in navigation.get("branches") or []:
        path_selection = branch.get("path_selection")
        if isinstance(path_selection, dict):
            path_selection = {
                **path_selection,
                "attempt_id": attempt_id,
                "selected_paths": [
                    {**path, "attempt_id": attempt_id}
                    for path in path_selection.get("selected_paths") or []
                ],
            }
        tagged_branches.append(
            {**branch, "attempt_id": attempt_id, "path_selection": path_selection}
        )
    tagged_navigation = {**navigation, "attempt_id": attempt_id, "branches": tagged_branches}
    best_path_stage = trace.get("best_path_stage") or {}
    tagged_best_path_stage = {
        **best_path_stage,
        "attempt_id": attempt_id,
        "selected_paths": [
            {**path, "attempt_id": attempt_id}
            for path in best_path_stage.get("selected_paths") or []
        ],
    }
    tagged_memories = [
        {
            **memory,
            "retrieval_attempt_id": attempt_id,
            "graph_paths": [
                {**path, "attempt_id": attempt_id} for path in memory.get("graph_paths") or []
            ],
        }
        for memory in result.memories
    ]
    return RetrievalResult(
        memories=tagged_memories,
        trace={
            **trace,
            "retrieval_attempt_id": attempt_id,
            "initial_embedding_bm25_rrf_stage": tagged_seed_stage,
            "react_path_navigation": tagged_navigation,
            "best_path_stage": tagged_best_path_stage,
        },
    )


def _post_path_assessment_input(result: RetrievalResult) -> dict:
    """Describe the answer-visible node union produced by one complete-path attempt."""
    trace = result.trace
    seed_stage = trace.get("initial_embedding_bm25_rrf_stage") or {}
    best_path_stage = trace.get("best_path_stage") or {}
    path_budget_stage = trace.get("path_budget_stage") or {}
    assessed_paths = (
        path_budget_stage.get("selected_paths") or []
        if path_budget_stage.get("enabled")
        else best_path_stage.get("selected_paths") or []
    )
    child_fact_ids = [
        str(fact.get("id") or "")
        for memory in result.memories
        for fact in memory.get("query_relevant_child_facts") or []
        if str(fact.get("id") or "")
    ]
    return {
        "attempt_id": str(trace.get("retrieval_attempt_id") or ""),
        "seed_ids": [
            str(item.get("id") or "")
            for item in seed_stage.get("selected") or []
            if str(item.get("id") or "")
        ],
        "selected_path_ids": [
            str(path.get("path_id") or "")
            for path in assessed_paths
            if str(path.get("path_id") or "")
        ],
        "node_ids": [str(memory.get("id") or "") for memory in result.memories],
        "child_fact_ids": child_fact_ids,
        "child_fact_count": len(child_fact_ids),
        "path_integrity": "budget_selected_complete_paths_plus_reranked_zero_hop_seeds"
        if path_budget_stage.get("enabled")
        else "controller_selected_complete_paths_plus_reranked_zero_hop_seeds",
    }


def _post_path_attempt_trace(
    result: RetrievalResult, *, attempt_id: str, rrf_candidate_top_n: int, rerank_top_n: int
) -> dict:
    trace = result.trace
    navigation = trace.get("react_path_navigation") or {}
    return {
        "attempt_id": attempt_id,
        "rrf_candidate_top_n": int(rrf_candidate_top_n),
        "rerank_top_n": int(rerank_top_n),
        "selected_memory_count": len(result.memories),
        "selected_memory_ids": [str(memory.get("id") or "") for memory in result.memories],
        "seed_stage": trace.get("initial_embedding_bm25_rrf_stage") or {},
        "navigation": navigation,
        "best_path_stage": trace.get("best_path_stage") or {},
        "controller_prompt_tokens": int(navigation.get("controller_prompt_tokens") or 0),
    }


def _apply_complete_path_budget_and_child_facts(
    question: str, scope: dict, result: RetrievalResult, *, mode: str, max_nodes: int | None = None
) -> RetrievalResult:
    """Apply complete-path budgeting once, then attach child facts to its Mid union."""
    effective_soft_max_nodes = effective_hard_max_nodes = int(max_nodes)
    memory_by_id = {
        str(memory.get("id") or ""): memory
        for memory in result.memories
        if str(memory.get("id") or "")
    }
    if not memory_by_id:
        omitted_stages = {
            "path_budget_stage",
            "child_atomic_evidence_stage",
            "source_dialogue_stage",
            "final_rerank_stage",
            "final",
        }
        base_trace = {
            key: value for key, value in result.trace.items() if key not in omitted_stages
        }
        return RetrievalResult(
            memories=[],
            trace={
                **base_trace,
                "mode": mode,
                "resolved_parameters": _resolved_parameter_trace(
                    base_trace, complete_path_budget=effective_hard_max_nodes
                ),
                "path_budget_stage": {
                    "enabled": True,
                    "unit": "complete_path_bundle",
                    "max_unique_nodes": effective_hard_max_nodes,
                    "soft_max_unique_nodes": effective_soft_max_nodes,
                    "hard_max_unique_nodes": effective_hard_max_nodes,
                    "candidate_path_count": 0,
                    "selected_path_count": 0,
                    "selected_unique_node_count": 0,
                    "reason": "empty_post_path_graph_result",
                },
                "child_atomic_evidence_stage": {
                    "enabled": True,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "selected_mid_count": 0,
                    "reason": "empty_post_path_graph_result",
                },
                "final_rerank_stage": {"enabled": False, "candidate_count": 0, "selected_count": 0},
                "final": {
                    "mid_count": 0,
                    "core_count": 0,
                    "other_count": 0,
                    "long_count": 0,
                    "child_long_fact_count": 0,
                    "mids_with_child_facts": 0,
                    "count": 0,
                    "selected": [],
                    "path_integrity": "no_complete_path_candidates",
                    "path_node_budget": effective_hard_max_nodes,
                    "path_node_soft_budget": effective_soft_max_nodes,
                    "path_node_hard_budget": effective_hard_max_nodes,
                    "reason": "empty_post_path_graph_result",
                },
            },
        )
    trace = result.trace
    active_attempt_id = str(trace.get("retrieval_attempt_id") or "")
    seed_stage = trace.get("initial_embedding_bm25_rrf_stage") or {}
    best_path_stage = trace.get("best_path_stage") or {}
    strategy = trace.get("question_strategy") or {}
    seed_items = list(seed_stage.get("selected") or [])
    attempt_global_seed_ids = frozenset(
        str(item.get("id") or "") for item in seed_items if str(item.get("id") or "")
    )
    candidates = [
        {**candidate, "attempt_id": str(candidate.get("attempt_id") or active_attempt_id)}
        for candidate in complete_path_budget_candidates(
            best_path_stage.get("selected_paths") or [], seed_stage.get("selected") or []
        )
    ]
    budget_kwargs = {"max_nodes": int(max_nodes)}
    scoring_kwargs = {
        **budget_kwargs,
        "preferred_relation_types": strategy.get("preferred_relation_types") or [],
        "rerank_scored_fn": rerank_scored,
        "path_score_weights": (
            float(config.GRAPH_PATH_WEIGHT_NODE),
            float(config.GRAPH_PATH_WEIGHT_TYPE),
            float(config.GRAPH_PATH_WEIGHT_DESC),
            float(config.GRAPH_PATH_WEIGHT_CONTROLLER),
        ),
    }
    selection = score_and_select_complete_paths(
        question, candidates, memory_by_id, **scoring_kwargs
    )
    selected_paths = selection["selected_paths"]
    selected_node_ids = selection["selected_node_ids"]
    node_annotations = selection["node_annotations"]
    prebudget_non_seed_node_ids = [
        node_id for node_id in memory_by_id if node_id not in attempt_global_seed_ids
    ]
    selected_non_seed_node_ids = [
        node_id for node_id in selected_node_ids if node_id not in attempt_global_seed_ids
    ]
    selected_node_id_set = set(selected_node_ids)
    removed_non_seed_node_ids = [
        node_id for node_id in prebudget_non_seed_node_ids if node_id not in selected_node_id_set
    ]
    seed_rank_by_id = {
        str(item.get("id") or ""): int(item.get("rerank_rank") or position)
        for position, item in enumerate(seed_items, start=1)
        if str(item.get("id") or "")
    }
    provenance_by_id: dict[str, list[dict]] = {}
    for node_id in selected_node_ids:
        if node_id not in seed_rank_by_id:
            continue
        base_paths = list(memory_by_id[node_id].get("graph_paths") or [])
        canonical = next(
            (
                item
                for item in base_paths
                if item.get("path_id") is None and str(item.get("seed_id") or "") == node_id
            ),
            None,
        )
        provenance_by_id[node_id] = [
            dict(canonical)
            if canonical is not None
            else {
                "seed_id": node_id,
                "seed_rank": seed_rank_by_id[node_id],
                "path_id": None,
                "attempt_id": active_attempt_id,
                "position": 0,
                "hop": 0,
                "path": [node_id],
                "value": None,
                "reason": "RRF seed retained independently of graph expansion",
            }
        ]
    for path in selected_paths:
        seed_id = str(path.get("seed_id") or (path.get("node_ids") or [""])[0])
        traversal_path = list(path.get("path") or path.get("node_ids") or [])
        for position, raw_node_id in enumerate(path.get("node_ids") or []):
            node_id = str(raw_node_id)
            provenance_by_id.setdefault(node_id, []).append(
                {
                    "seed_id": seed_id,
                    "seed_rank": seed_rank_by_id.get(seed_id),
                    "path_id": path.get("path_id"),
                    "attempt_id": str(path.get("attempt_id") or active_attempt_id),
                    "position": position,
                    "hop": position,
                    "path": traversal_path,
                    "value": path.get("value"),
                    "reason": path.get("reason"),
                    "path_budget_score": path.get("path_score"),
                }
            )
    child_facts_by_mid, child_fact_trace = _query_relevant_child_facts(
        question,
        selected_node_ids,
        scope,
        min_score=config.GRAPH_CHILD_FACT_MIN_SCORE,
        global_limit=int(config.GRAPH_CHILD_FACT_GLOBAL_TOP_N),
    )
    memories: list[dict] = []
    for node_id in selected_node_ids:
        base = memory_by_id[node_id]
        provenance = provenance_by_id[node_id]
        child_facts = child_facts_by_mid.get(node_id, [])
        memories.append(
            {
                **base,
                **node_annotations[node_id],
                "content": _mid_content_with_child_facts(base, child_facts),
                "memory_pool": "mid",
                "graph_hop": min((int(item.get("hop") or 0) for item in provenance)),
                "graph_path": provenance[0]["path"],
                "graph_paths": provenance,
                "query_relevant_child_facts": child_facts,
            }
        )
    path_budget_stage = {
        **selection["stage"],
        "candidate_source": "controller_selected_paths_plus_reranked_seed_zero_hop_paths",
        "seed_budget_policy": "all_reranked_seeds_compete_as_zero_hop_complete_paths",
        "child_fact_rerank_order": "after_path_budget",
        "attempt_seed_ids": sorted(attempt_global_seed_ids),
        "prebudget_unique_non_seed_node_count": len(prebudget_non_seed_node_ids),
        "prebudget_non_seed_node_ids": prebudget_non_seed_node_ids,
        "selected_unique_non_seed_node_count": len(selected_non_seed_node_ids),
        "selected_non_seed_node_ids": selected_non_seed_node_ids,
        "removed_unique_non_seed_node_count": len(removed_non_seed_node_ids),
        "removed_non_seed_node_ids": removed_non_seed_node_ids,
    }
    best_path_stage = {
        **best_path_stage,
        "seed_retention_policy": "all_rrf_seeds_enter_path_budget_as_zero_hop_candidates",
        "downstream_final_selection_stage": "path_budget_stage",
        "controller_paths_are_budget_candidates": True,
        "enumerated_paths_are_budget_candidates": False,
    }
    omitted_stages = {
        "path_budget_stage",
        "child_atomic_evidence_stage",
        "source_dialogue_stage",
        "final_rerank_stage",
        "final",
    }
    base_trace = {key: value for key, value in trace.items() if key not in omitted_stages}
    base_trace["best_path_stage"] = best_path_stage
    post_path_stage = base_trace.get("post_path_sufficiency_fallback")
    if isinstance(post_path_stage, dict):
        assessed_node_ids = list(
            (post_path_stage.get("assessment_input") or {}).get("node_ids") or []
        )
        active_prebudget_node_ids = list(memory_by_id)
        selected_set = set(selected_node_ids)
        post_path_stage["active_prebudget_node_ids"] = active_prebudget_node_ids
        post_path_stage["final_budget_node_ids"] = list(selected_node_ids)
        post_path_stage["active_node_ids_surviving_budget"] = [
            node_id for node_id in active_prebudget_node_ids if node_id in selected_set
        ]
        post_path_stage["active_node_ids_removed_by_budget"] = [
            node_id for node_id in active_prebudget_node_ids if node_id not in selected_set
        ]
        post_path_stage["assessed_node_ids_in_final_budget"] = [
            node_id for node_id in assessed_node_ids if node_id in selected_set
        ]
        post_path_stage["assessed_node_ids_absent_from_final_budget"] = [
            node_id for node_id in assessed_node_ids if node_id not in selected_set
        ]
        post_path_stage["assessment_attempt_matches_budget_attempt"] = str(
            (post_path_stage.get("assessment_input") or {}).get("attempt_id") or ""
        ) == str(post_path_stage.get("active_attempt_id") or "")
    hop_counts = Counter((memory["graph_hop"] for memory in memories))
    final_counts = {f"hop{hop}_mid_count": hop_counts[hop] for hop in range(6)}
    final_trace = {
        **base_trace,
        "mode": mode,
        "resolved_parameters": _resolved_parameter_trace(
            base_trace, complete_path_budget=effective_hard_max_nodes
        ),
        "path_budget_stage": path_budget_stage,
        "child_atomic_evidence_stage": child_fact_trace,
        "final_rerank_stage": {
            "enabled": False,
            "candidate_count": len(memories),
            "selected_count": len(memories),
        },
        "final": {
            **final_counts,
            "mid_count": len(memories),
            "core_count": 0,
            "other_count": 0,
            "long_count": 0,
            "child_long_fact_count": int(child_fact_trace.get("selected_count") or 0),
            "mids_with_child_facts": int(child_fact_trace.get("selected_mid_count") or 0),
            "count": len(memories),
            "selected": [memory["id"] for memory in memories],
            "unique_non_seed_mid_count": len(selected_non_seed_node_ids),
            "non_seed_mid_ids": selected_non_seed_node_ids,
            "path_integrity": "every_selected_path_is_complete",
            "path_node_budget": effective_hard_max_nodes,
            "path_node_soft_budget": effective_soft_max_nodes,
            "path_node_hard_budget": effective_hard_max_nodes,
            "prompt_projection": {
                "mid": "complete paths selected under a global unique-node budget: topic_subject + summary -> content",
                "relation_type_direction_description": "navigation_and_trace_only",
                "typed_long": "query-relevant child facts with sourceEvidence embedded under owning mid",
            },
        },
        "selected_graph_relations": _selected_graph_relations(selected_paths, selected_node_ids),
    }
    return RetrievalResult(memories=memories, trace=final_trace)


def _search_paths_with_post_path_gate(
    question: str,
    scope: dict,
    *,
    top_k: int | None = None,
    mode: str,
    max_hops_override: int,
    complete_path_max_unique_nodes: int = 10,
) -> RetrievalResult:
    """Judge the initial complete-path node union, optionally navigate wider, then budget and attach facts."""
    if int(complete_path_max_unique_nodes) <= 0:
        raise ValueError("complete-path Mid budget must be positive")
    initial_seed_top_n = int(config.GRAPH_INITIAL_SEED_TOP_N)
    initial_attempt_id = f"initial_top{initial_seed_top_n}"
    initial_rrf_top_n = int(config.GRAPH_INITIAL_RRF_CANDIDATE_TOP_N)
    fallback_seed_top_n = int(config.GRAPH_FALLBACK_SEED_TOP_N)
    if fallback_seed_top_n <= initial_seed_top_n:
        raise ValueError("fallback seed width must exceed the initial seed width")
    fallback_rrf_top_n = int(config.GRAPH_FALLBACK_RRF_CANDIDATE_TOP_N)
    if initial_rrf_top_n < initial_seed_top_n:
        raise ValueError("initial RRF candidate width cannot be below seed width")
    if fallback_rrf_top_n < fallback_seed_top_n:
        raise ValueError("fallback RRF candidate width cannot be below seed width")

    def apply_budget(result: RetrievalResult) -> RetrievalResult:
        return _apply_complete_path_budget_and_child_facts(
            question, scope, result, mode=mode, max_nodes=int(complete_path_max_unique_nodes)
        )

    initial_graph = _tag_post_path_attempt(
        _run_one_path_attempt(
            question,
            scope,
            top_k=top_k,
            mode=mode,
            seed_top_n=initial_seed_top_n,
            rrf_candidate_top_n=initial_rrf_top_n,
            max_hops_override=max_hops_override,
        ),
        attempt_id=initial_attempt_id,
    )
    assessment_input = _post_path_assessment_input(initial_graph)
    if initial_graph.memories:
        assessment = judge_seed_set_sufficiency(question, initial_graph.memories)
    else:
        assessment = {
            "sufficient": False,
            "missing": "The initial complete-path evidence set is empty.",
            "model": None,
            "attempts": 0,
            "prompt_tokens": 0,
        }
    fallback_triggered = not bool(assessment["sufficient"])
    fallback_attempt_id = f"fallback_top{fallback_seed_top_n}"
    if fallback_triggered:
        final_graph = _tag_post_path_attempt(
            _run_one_path_attempt(
                question,
                scope,
                top_k=top_k,
                mode=mode,
                seed_top_n=fallback_seed_top_n,
                rrf_candidate_top_n=fallback_rrf_top_n,
                max_hops_override=max_hops_override,
            ),
            attempt_id=fallback_attempt_id,
        )
    else:
        final_graph = initial_graph
    initial_attempt = _post_path_attempt_trace(
        initial_graph,
        attempt_id=initial_attempt_id,
        rrf_candidate_top_n=initial_rrf_top_n,
        rerank_top_n=initial_seed_top_n,
    )
    fallback_attempt = (
        _post_path_attempt_trace(
            final_graph,
            attempt_id=fallback_attempt_id,
            rrf_candidate_top_n=fallback_rrf_top_n,
            rerank_top_n=fallback_seed_top_n,
        )
        if fallback_triggered
        else {
            "attempt_id": fallback_attempt_id,
            "executed": False,
            "rrf_candidate_top_n": None,
            "rerank_top_n": None,
            "selected_memory_count": 0,
            "selected_memory_ids": [],
            "controller_prompt_tokens": 0,
        }
    )
    fallback_attempt["executed"] = fallback_triggered
    controller_prompt_tokens = int(initial_attempt["controller_prompt_tokens"])
    if fallback_triggered:
        controller_prompt_tokens += int(fallback_attempt["controller_prompt_tokens"])
    sufficiency_prompt_tokens = int((assessment or {}).get("prompt_tokens") or 0)
    final_graph = RetrievalResult(
        memories=final_graph.memories,
        trace={
            **final_graph.trace,
            "mode": mode,
            "post_path_sufficiency_fallback": {
                "enabled": True,
                "configuration": {
                    "max_hops": int(max_hops_override),
                    "max_paths_per_seed": int(config.GRAPH_MAX_PATHS_PER_SEED),
                    "initial_seed_top_n": initial_seed_top_n,
                    "initial_rrf_candidate_top_n": initial_rrf_top_n,
                    "fallback_seed_top_n": fallback_seed_top_n,
                    "fallback_rrf_candidate_top_n": fallback_rrf_top_n,
                },
                "triggered": fallback_triggered,
                "assessment_stage": "after_initial_graph_navigation_and_complete_path_selection_before_complete_path_rerank_and_budget",
                "assessment_projection": "ordered_unique_complete_path_node_union_topic_plus_summary",
                "child_facts_before_sufficiency": False,
                "assessment_input": assessment_input,
                "assessment": assessment,
                "active_attempt_id": fallback_attempt_id
                if fallback_triggered
                else initial_attempt_id,
                "initial_attempt": initial_attempt,
                "fallback_attempt": fallback_attempt,
                "attempt_count": 2 if fallback_triggered else 1,
                "prompt_tokens": {
                    "graph_navigation_and_path_selection": controller_prompt_tokens,
                    "sufficiency": sufficiency_prompt_tokens,
                    "total": controller_prompt_tokens + sufficiency_prompt_tokens,
                },
            },
        },
    )
    return apply_budget(final_graph)


def search_graph_paths(question: str, scope: dict, *, top_k: int | None = None) -> RetrievalResult:
    """Retrieve memories for `question` by navigating the mid-memory relation graph.

    `scope` supplies the graph to walk: `mids` (candidate mid memories), `mid_rels`
    (typed relations between them), and optionally `mid_by_id`. The walk seeds from a
    hybrid dense+BM25 ranking, lets one agent per seed choose relation paths, judges
    whether the collected evidence answers the question, and widens the seed set once if
    it does not. Widths, hop limit, and path budget come from `config`.
    """
    return _search_paths_with_post_path_gate(
        question,
        scope,
        top_k=top_k,
        mode="graph",
        max_hops_override=int(config.GRAPH_MAX_HOPS),
        complete_path_max_unique_nodes=int(config.GRAPH_COMPLETE_PATH_MAX_UNIQUE_NODES),
    )


__all__ = ["clear_enriched_embedding_cache", "search_graph_paths"]
