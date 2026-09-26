

from __future__ import annotations

from .projections import mid_content


def edge_cards(
    frontier_ids: set[str],
    visited_ids: set[str],
    evaluated_edge_ids: set[str],
    relations: list[dict],
    mid_by_id: dict[str, dict],
) -> list[dict]:
    """Build undirected, unvisited frontier-edge cards for one navigation round."""
    cards: list[dict] = []
    for edge in relations:
        edge_id = str(edge.get("id") or "")
        if not edge_id or edge_id in evaluated_edge_ids:
            continue
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        if source_id in frontier_ids and target_id not in visited_ids:
            current_id, neighbour_id = source_id, target_id
        elif target_id in frontier_ids and source_id not in visited_ids:
            current_id, neighbour_id = target_id, source_id
        else:
            continue
        current = mid_by_id.get(current_id)
        neighbour = mid_by_id.get(neighbour_id)
        if current is None or neighbour is None or not mid_content(neighbour):
            continue
        description = " ".join(str(edge.get("description") or "").split())
        if not description:
            raise ValueError(
                "graph navigation requires every traversable edge to have a "
                f"description (missing on {edge_id})"
            )
        cards.append(
            {
                "edge_id": edge_id,
                "source_id": source_id,
                "target_id": target_id,
                "current_node_id": current_id,
                "current_topic": current.get("topic_subject") or "",
                "neighbour_id": neighbour_id,
                "neighbour_topic": neighbour.get("topic_subject") or "",
                "relation_type": edge.get("relation_type"),
                "description": description,
            }
        )
    return sorted(cards, key=lambda card: card["edge_id"])


def new_branch(seed: dict, seed_rank: int) -> dict:
    seed_id = str(seed["id"])
    return {
        "seed_id": seed_id,
        "seed_rank": seed_rank,
        "visited": {seed_id: seed},
        "frontier_ids": {seed_id},
        "evaluated_edge_ids": set(),
        "hop_by_id": {seed_id: 0},
        "paths_by_id": {seed_id: [[seed_id]]},
        "rounds": [],
        "active": True,
        "stop_reason": None,
    }


def branch_records(branch: dict) -> list[dict]:
    """Return one branch's records in deterministic hop/discovery order."""
    return list(branch["visited"].values())


def _dedupe_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def apply_branch_decision(
    branch: dict,
    cards: list[dict],
    decision: dict,
    mid_by_id: dict[str, dict],
    hop: int,
) -> tuple[list[str], dict]:
    card_by_id = {card["edge_id"]: card for card in cards}
    selected_cards = [card_by_id[edge_id] for edge_id in decision["selected_edge_ids"]]
    branch["evaluated_edge_ids"].update(card_by_id)
    round_trace = {
        "expansion_hop": hop,
        "frontier_node_ids": sorted(branch["frontier_ids"]),
        "memory_count_before": len(branch["visited"]),
        "candidate_edge_count": len(cards),
        "candidate_edges": cards,
        "decision": decision,
        "selected_edges": [],
        "expanded_node_ids": [],
    }

    for card in selected_cards:
        current = mid_by_id.get(card["current_node_id"], {})
        neighbour = mid_by_id.get(card["neighbour_id"], {})
        round_trace["selected_edges"].append(
            {
                **card,
                "cross_user": bool(
                    current.get("user_id")
                    and neighbour.get("user_id")
                    and current.get("user_id") != neighbour.get("user_id")
                ),
            }
        )

    if decision["sufficient"]:
        branch["active"] = False
        branch["stop_reason"] = f"locally_sufficient_before_hop_{hop}"
        round_trace["memory_count_after"] = len(branch["visited"])
        branch["rounds"].append(round_trace)
        return [], round_trace
    if not selected_cards:
        branch["active"] = False
        branch["stop_reason"] = "no_selected_edges" if cards else "no_candidate_edges"
        round_trace["memory_count_after"] = len(branch["visited"])
        branch["rounds"].append(round_trace)
        return [], round_trace

    pending_paths: dict[str, list[list[str]]] = {}
    for card in selected_cards:
        neighbour_id = card["neighbour_id"]
        if neighbour_id in branch["visited"]:
            continue
        neighbour = mid_by_id.get(neighbour_id)
        if neighbour is None or not mid_content(neighbour):
            continue
        current_paths = branch["paths_by_id"].get(
            card["current_node_id"], [[card["current_node_id"]]]
        )
        pending_paths.setdefault(neighbour_id, []).extend(
            [*path, card["edge_id"], neighbour_id] for path in current_paths
        )

    new_frontier: list[str] = []
    for neighbour_id, paths in pending_paths.items():
        neighbour = mid_by_id[neighbour_id]
        branch["visited"][neighbour_id] = neighbour
        branch["hop_by_id"][neighbour_id] = hop
        branch["paths_by_id"][neighbour_id] = _dedupe_paths(paths)
        new_frontier.append(neighbour_id)

    branch["frontier_ids"] = set(new_frontier)
    round_trace["expanded_node_ids"] = new_frontier
    round_trace["memory_count_after"] = len(branch["visited"])
    if not new_frontier:
        branch["active"] = False
        branch["stop_reason"] = "selected_edges_added_no_nodes"
    branch["rounds"].append(round_trace)
    return new_frontier, round_trace
